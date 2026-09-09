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
# Standard SD sampling defaults — the safe choice for a normal (non-distilled)
# checkpoint, which is what most users have loaded. At low step counts these
# checkpoints don't have time to converge and produce noisy, incoherent
# output rather than merely "less detail".
_DEFAULT_STEPS = 20
_DEFAULT_CFG_SCALE = 7.0
# Distilled "Turbo"/"Lightning"/LCM checkpoints are trained for 1-4 step
# sampling at a near-1 CFG scale — the standard defaults above way overcook
# them into a blown-out, distorted mess. Auto-selected via
# _pick_sampling_defaults() when the checkpoint filename says so (currently
# only possible for ComfyUI, which takes an explicit checkpoint filename).
_DISTILLED_STEPS = 4
_DISTILLED_CFG_SCALE = 1.5
_DISTILLED_CHECKPOINT_MARKERS = ("turbo", "lightning", "lcm")
_DEFAULT_NEGATIVE_PROMPT = "text, watermark, signature, low quality, blurry"
_COMFYUI_POLL_INTERVAL = 1.0
_SEED_MAX = 2**32 - 1

# SDXL-family checkpoints are trained at ~1024px and produce a tiled,
# "shattered glass" mess when sampled at the 512px SD1.5 default — a
# different failure mode from the steps/cfg mismatch above, and one that
# also needs a much bigger base resolution to fix, not just more steps.
_SDXL_CHECKPOINT_MARKERS = ("xl",)

# Base generation resolution per _AddonConfig.local_image_resolution tier,
# by aspect hint. Local diffusion models want dimensions close to what they
# were trained at — SD1.5-family checkpoints are happiest around 512-768px,
# SDXL-family ones need ~1024px, or output degrades into a tiled/fractured
# look rather than just "less detail".
_SIZE_TIERS: dict[str, dict[str, tuple[int, int]]] = {
    "512": {"auto": (512, 512), "square": (512, 512), "portrait": (512, 768), "landscape": (768, 512)},
    "768": {"auto": (768, 768), "square": (768, 768), "portrait": (768, 1152), "landscape": (1152, 768)},
    "1024": {"auto": (1024, 1024), "square": (1024, 1024), "portrait": (896, 1152), "landscape": (1152, 896)},
}


def _resolve_size(size: str | None, resolution: str = "512") -> tuple[int, int]:
    """Map a plugin-wide size hint + resolution tier to (width, height) pixels.

    Args:
        size: Aspect hint ("auto", "square", "portrait", "landscape").
        resolution: Base resolution tier ("512", "768", or "1024") — see
            _SIZE_TIERS. Falls back to "512" for an unrecognized value.
    """
    tier = _SIZE_TIERS.get(resolution, _SIZE_TIERS["512"])
    return tier.get(size or "auto", tier["auto"])


def _looks_sdxl_checkpoint(checkpoint: str) -> bool:
    """True if a checkpoint filename looks like an SDXL-family model.

    Matches "xl" as a standalone token (e.g. 'sd_xl_turbo', 'sdxl_base',
    'juggernaut-xl') so it doesn't false-positive on unrelated substrings.
    """
    import re

    tokens = re.split(r"[^a-z0-9]+", checkpoint.lower())
    return "xl" in tokens or any(t.endswith("xl") and len(t) > 2 for t in tokens)


def _pick_sampling_defaults(checkpoint: str) -> tuple[int, float]:
    """Pick (steps, cfg_scale) based on whether a checkpoint name looks distilled.

    Args:
        checkpoint: Checkpoint filename, e.g. 'sd_xl_turbo_1.0.safetensors'.

    Returns:
        (steps, cfg_scale) — the low-step distilled-model defaults if the
        name contains a marker like "turbo"/"lightning"/"lcm", otherwise the
        standard SD defaults.
    """
    name = checkpoint.lower()
    if any(marker in name for marker in _DISTILLED_CHECKPOINT_MARKERS):
        return _DISTILLED_STEPS, _DISTILLED_CFG_SCALE
    return _DEFAULT_STEPS, _DEFAULT_CFG_SCALE


def _resolve_resolution(setting: str, *, checkpoint: str | None) -> str:
    """Resolve the AddonConfig.local_image_resolution setting to a size tier.

    Args:
        setting: "auto", "512", "768", or "1024".
        checkpoint: Checkpoint filename for auto-detection (ComfyUI only —
            Automatic1111 has no checkpoint info to detect from, so pass
            None there and "auto" falls back to "512").

    Returns:
        A key into _SIZE_TIERS ("512", "768", or "1024").
    """
    if setting in _SIZE_TIERS:
        return setting
    if checkpoint and _looks_sdxl_checkpoint(checkpoint):
        return "1024"
    return "512"


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
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        resolution: str = "512",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._resolution = resolution
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
        width, height = _resolve_size(size, self._resolution)
        body = {
            "prompt": prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "cfg_scale": self._cfg_scale,
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
    cfg: float,
    seed: int,
) -> dict[str, object]:
    """Build a minimal txt2img node graph: checkpoint -> CLIP encode -> KSampler -> VAE decode -> save."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
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
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        resolution: str = "512",
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
        self._cfg_scale = cfg_scale
        self._resolution = resolution
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
        width, height = _resolve_size(size, self._resolution)
        seed = random.randint(0, _SEED_MAX)  # noqa: S311
        workflow = _build_comfyui_workflow(
            prompt, self._negative_prompt, self._checkpoint, width, height, self._steps, self._cfg_scale, seed
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
        # No checkpoint info is available for Automatic1111 (it just uses
        # whatever's loaded in the WebUI), so "auto" can't detect SDXL here —
        # falls back to the SD1.5-safe 512 tier unless overridden in Settings.
        resolution = _resolve_resolution(config.local_image_resolution, checkpoint=None)
        return Automatic1111Client(config.local_image_url, resolution=resolution)

    if config.local_image_backend == "comfyui":
        checkpoint = config.local_image_checkpoint.strip()
        if not checkpoint:
            return None
        steps, cfg_scale = _pick_sampling_defaults(checkpoint)
        resolution = _resolve_resolution(config.local_image_resolution, checkpoint=checkpoint)
        return ComfyUIClient(
            config.local_image_url, checkpoint, steps=steps, cfg_scale=cfg_scale, resolution=resolution
        )

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
