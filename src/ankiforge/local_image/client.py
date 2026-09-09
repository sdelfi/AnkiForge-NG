"""Local image generation backends: Automatic1111 (stable-diffusion-webui) and ComfyUI.

Unlike OpenRouter and OpenAI-compatible chat servers (LM Studio, Ollama, vLLM), these
two don't speak the /chat/completions "modalities" format that
ankiforge.openrouter.client.OpenRouterClient.generate_image() relies on — they're
purpose-built image generation APIs with their own request/response shapes. So they
get their own clients here, wired into RoutingClient as a third route (see
ankiforge.openrouter.routing_client.LOCAL_IMAGE_PREFIX) alongside OpenRouter and the
custom OpenAI-compatible endpoint.
"""

from __future__ import annotations

import base64
import random
import time
from typing import TYPE_CHECKING, Protocol

import requests

if TYPE_CHECKING:
    from ankiforge.models import AddonConfig

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_STEPS = 20
_DEFAULT_NEGATIVE_PROMPT = "text, watermark, signature, low quality, blurry"
_COMFYUI_POLL_INTERVAL = 1.0
_SEED_MAX = 2**32 - 1

# Rough size presets — local diffusion models are usually happiest at 512-768px
# per side; there's no per-provider "size" enum to map onto like OpenRouter has.
_SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "auto": (512, 512),
    "square": (512, 512),
    "portrait": (512, 768),
    "landscape": (768, 512),
}


def _resolve_size(size: str | None) -> tuple[int, int]:
    """Map a plugin-wide size hint to (width, height) pixels."""
    return _SIZE_PRESETS.get(size or "auto", _SIZE_PRESETS["auto"])


class LocalImageError(Exception):
    """Raised when a local image generation backend fails."""


class LocalImageClient(Protocol):
    """Interface RoutingClient depends on for local image generation."""

    def generate_image(self, prompt: str, *, size: str | None = None) -> bytes:
        """Generate an image and return its raw bytes (PNG)."""
        ...


class Automatic1111Client:
    """Client for the AUTOMATIC1111 stable-diffusion-webui REST API.

    Uses whatever checkpoint is currently loaded in the WebUI — there's no
    per-request model selection, unlike OpenRouter/ComfyUI.
    """

    def __init__(
        self,
        base_url: str,
        *,
        steps: int = _DEFAULT_STEPS,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._steps = steps
        self._negative_prompt = negative_prompt
        self._timeout = timeout

    def generate_image(self, prompt: str, *, size: str | None = None) -> bytes:
        """Generate an image via POST /sdapi/v1/txt2img.

        Args:
            prompt: Image description.
            size: Plugin-wide size hint (mapped to width/height).

        Returns:
            PNG image bytes.

        Raises:
            LocalImageError: On a network error, non-200 response, or a
                response missing image data.
        """
        width, height = _resolve_size(size)
        body = {
            "prompt": prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "width": width,
            "height": height,
        }
        try:
            resp = requests.post(f"{self.base_url}/sdapi/v1/txt2img", json=body, timeout=self._timeout)
        except Exception as e:  # noqa: BLE001
            msg = f"Network error contacting Automatic1111: {e}"
            raise LocalImageError(msg) from e

        if resp.status_code != 200:
            msg = f"Automatic1111 returned status {resp.status_code}: {resp.text[:300]}"
            raise LocalImageError(msg)

        try:
            images = resp.json()["images"]
        except (ValueError, KeyError) as e:
            msg = "Automatic1111 response missing image data"
            raise LocalImageError(msg) from e

        if not images:
            msg = "Automatic1111 returned no images"
            raise LocalImageError(msg)

        return base64.b64decode(images[0])


def _build_comfyui_workflow(
    prompt: str,
    negative_prompt: str,
    checkpoint: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    """Build a minimal txt2img node graph: checkpoint -> CLIP encode -> KSampler -> VAE decode -> save."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": steps,
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ankiforge", "images": ["8", 0]}},
    }


class ComfyUIClient:
    """Client for the ComfyUI REST API.

    Submits a generated txt2img workflow to POST /prompt, polls GET
    /history/{id} until the run completes, then fetches the output image via
    GET /view. ComfyUI has no default checkpoint like Automatic1111 does, so
    the checkpoint (.safetensors) filename must be provided explicitly.
    """

    def __init__(
        self,
        base_url: str,
        checkpoint: str,
        *,
        steps: int = _DEFAULT_STEPS,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        timeout: float = _DEFAULT_TIMEOUT,
        poll_interval: float = _COMFYUI_POLL_INTERVAL,
    ) -> None:
        if not checkpoint.strip():
            msg = "ComfyUI requires a checkpoint filename — set it in AnkiForge Settings"
            raise ValueError(msg)
        self.base_url = base_url.rstrip("/")
        self._checkpoint = checkpoint.strip()
        self._steps = steps
        self._negative_prompt = negative_prompt
        self._timeout = timeout
        self._poll_interval = poll_interval

    def generate_image(self, prompt: str, *, size: str | None = None) -> bytes:
        """Generate an image by submitting and polling a ComfyUI workflow.

        Args:
            prompt: Image description.
            size: Plugin-wide size hint (mapped to width/height).

        Returns:
            Raw image bytes, as returned by ComfyUI's /view endpoint.

        Raises:
            LocalImageError: On a network error, a non-200 response, or a
                timeout waiting for generation to finish.
        """
        width, height = _resolve_size(size)
        seed = random.randint(0, _SEED_MAX)  # noqa: S311
        workflow = _build_comfyui_workflow(
            prompt, self._negative_prompt, self._checkpoint, width, height, self._steps, seed
        )

        try:
            resp = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30)
        except Exception as e:  # noqa: BLE001
            msg = f"Network error contacting ComfyUI: {e}"
            raise LocalImageError(msg) from e

        if resp.status_code != 200:
            msg = f"ComfyUI returned status {resp.status_code}: {resp.text[:300]}"
            raise LocalImageError(msg)

        try:
            prompt_id = resp.json()["prompt_id"]
        except (ValueError, KeyError) as e:
            msg = "ComfyUI did not return a prompt_id"
            raise LocalImageError(msg) from e

        return self._wait_and_fetch(prompt_id)

    def _wait_and_fetch(self, prompt_id: str) -> bytes:
        """Poll /history/{prompt_id} until outputs appear, then fetch the image."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            except Exception as e:  # noqa: BLE001
                msg = f"Network error polling ComfyUI: {e}"
                raise LocalImageError(msg) from e

            if resp.status_code == 200:
                entry = resp.json().get(prompt_id)
                if entry and entry.get("outputs"):
                    return self._extract_image(entry["outputs"])

            time.sleep(self._poll_interval)

        msg = "Timed out waiting for ComfyUI to finish generating"
        raise LocalImageError(msg)

    def _extract_image(self, outputs: dict[str, object]) -> bytes:
        """Fetch the first image referenced in a completed workflow's outputs."""
        for node_output in outputs.values():
            images = node_output.get("images") if isinstance(node_output, dict) else None
            if images:
                img = images[0]
                params = {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
                try:
                    resp = requests.get(f"{self.base_url}/view", params=params, timeout=30)
                except Exception as e:  # noqa: BLE001
                    msg = f"Network error fetching image from ComfyUI: {e}"
                    raise LocalImageError(msg) from e
                if resp.status_code != 200:
                    msg = f"ComfyUI /view returned status {resp.status_code}"
                    raise LocalImageError(msg)
                return resp.content

        msg = "ComfyUI finished but produced no images"
        raise LocalImageError(msg)


def build_local_image_client(config: AddonConfig) -> LocalImageClient | None:
    """Build a local image client from add-on configuration, if one is set up.

    Args:
        config: Current add-on configuration.

    Returns:
        An Automatic1111Client or ComfyUIClient, or None if no local image
        backend is configured (backend + URL, and for ComfyUI, a checkpoint).
    """
    if not config.local_image_backend or not config.local_image_url.strip():
        return None

    if config.local_image_backend == "automatic1111":
        return Automatic1111Client(config.local_image_url)

    if config.local_image_backend == "comfyui":
        if not config.local_image_checkpoint.strip():
            return None
        return ComfyUIClient(config.local_image_url, config.local_image_checkpoint)

    return None


def validate_local_image_backend(backend: str, base_url: str) -> tuple[bool, str | None]:
    """Check that a local image generation backend is reachable.

    Args:
        backend: "automatic1111" or "comfyui".
        base_url: Base URL of the backend, e.g. 'http://127.0.0.1:7860'.

    Returns:
        Tuple (is_reachable, error_message). If reachable — (True, None).
    """
    if not base_url or not base_url.strip():
        return False, "Base URL is empty"

    url = base_url.strip().rstrip("/")
    health_path = "/sdapi/v1/sd-models" if backend == "automatic1111" else "/system_stats"

    try:
        resp = requests.get(f"{url}{health_path}", timeout=10)
    except Exception as e:  # noqa: BLE001
        return False, f"Network error: {e}"

    if resp.status_code == 200:
        return True, None

    return False, f"Unexpected response: {resp.status_code}"
