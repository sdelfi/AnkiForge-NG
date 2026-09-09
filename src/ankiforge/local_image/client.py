"""Local image generation backends: Automatic1111 (stable-diffusion-webui), Draw Things, and ComfyUI.

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
import json
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
_DEFAULT_SAMPLER = "euler"
# Automatic1111Client and DrawThingsClient both speak dropdown-style sampler
# names ("Euler", "Euler a", "DPM++ 2M Karras", ...) rather than ComfyUI's
# node-based lowercase_with_underscore ids. Draw Things defaults to the
# ancestral variant since it's commonly used with Turbo/Lightning-distilled
# checkpoints that need it for coherent few-step output (see _DISTILLED_SAMPLER
# above for the same reasoning on the ComfyUI side).
_DEFAULT_SAMPLER_A1111 = "Euler"
_DEFAULT_SAMPLER_DRAW_THINGS = "Euler a"
# Distilled "Turbo"/"Lightning"/LCM checkpoints are trained for 1-4 step
# sampling at a near-1 CFG scale — the standard defaults above way overcook
# them into a blown-out, distorted mess. Auto-selected via
# _pick_sampling_defaults() when the checkpoint filename says so (currently
# only possible for ComfyUI, which takes an explicit checkpoint filename).
#
# Sampler matters here as much as steps/cfg: plain "euler" is a deterministic
# sampler that expects a full-length step schedule to converge smoothly.
# Forced through only 1-4 steps it produces a fractured, tiled mess rather
# than a coherent (if soft) image — this is a *different* failure mode from
# the steps/cfg mismatch, and persists even at the correct resolution.
# Stability AI's own SDXL-Turbo model card recommends the ancestral variant
# (adds noise back in at each step) specifically for few-step sampling.
_DISTILLED_STEPS = 4
_DISTILLED_CFG_SCALE = 1.5
_DISTILLED_SAMPLER = "euler_ancestral"
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


def _pick_sampling_defaults(checkpoint: str) -> tuple[int, float, str]:
    """Pick (steps, cfg_scale, sampler_name) based on whether a checkpoint looks distilled.

    Args:
        checkpoint: Checkpoint filename, e.g. 'sd_xl_turbo_1.0.safetensors'.

    Returns:
        (steps, cfg_scale, sampler_name) — the low-step, ancestral-sampler
        distilled-model defaults if the name contains a marker like
        "turbo"/"lightning"/"lcm", otherwise the standard SD defaults.
    """
    name = checkpoint.lower()
    if any(marker in name for marker in _DISTILLED_CHECKPOINT_MARKERS):
        return _DISTILLED_STEPS, _DISTILLED_CFG_SCALE, _DISTILLED_SAMPLER
    return _DEFAULT_STEPS, _DEFAULT_CFG_SCALE, _DEFAULT_SAMPLER


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
    if checkpoint:
        # SDXL-Turbo specifically was distilled at 512x512, unlike base SDXL
        # or SDXL-Lightning (which target the full 1024x1024 resolution) —
        # confirmed empirically: this checkpoint produces a coherent image at
        # 512 and a tiled/fractured one at 1024, regardless of steps/sampler.
        if "turbo" in checkpoint.lower():
            return "512"
        if _looks_sdxl_checkpoint(checkpoint):
            return "1024"
    return "512"


# Advanced JSON override keys, and how to coerce each from the parsed JSON
# value. Keeping this as one small JSON escape hatch — rather than a combo
# box per parameter — covers checkpoints/samplers the auto-detection
# heuristics above don't recognize, without needing a code change each time.
_ADVANCED_OVERRIDE_KEYS = ("steps", "cfg_scale", "sampler_name", "resolution")


def _apply_advanced_overrides(
    steps: int, cfg_scale: float, sampler_name: str, resolution: str, raw_json: str
) -> tuple[int, float, str, str]:
    """Apply user-supplied JSON overrides on top of the auto-picked defaults.

    Args:
        steps: Auto-picked step count.
        cfg_scale: Auto-picked CFG scale.
        sampler_name: Auto-picked sampler.
        resolution: Auto-picked resolution tier.
        raw_json: Raw JSON object string, e.g.
            '{"steps": 8, "cfg_scale": 2, "sampler_name": "dpmpp_2m_sde"}'.
            Empty, invalid, or non-object JSON is ignored — falls back to
            the auto-picked values, untouched. Unrecognized or wrong-typed
            keys are ignored individually rather than failing the whole
            override; anything is a no-op that's always safe to leave in
            Settings.

    Returns:
        (steps, cfg_scale, sampler_name, resolution) with any valid
        overrides applied.
    """
    text = raw_json.strip()
    if not text:
        return steps, cfg_scale, sampler_name, resolution

    try:
        data = json.loads(text)
    except ValueError:
        return steps, cfg_scale, sampler_name, resolution
    if not isinstance(data, dict):
        return steps, cfg_scale, sampler_name, resolution

    raw_steps = data.get("steps")
    if isinstance(raw_steps, (int, float)) and not isinstance(raw_steps, bool):
        steps = int(raw_steps)

    raw_cfg = data.get("cfg_scale")
    if isinstance(raw_cfg, (int, float)) and not isinstance(raw_cfg, bool):
        cfg_scale = float(raw_cfg)

    raw_sampler = data.get("sampler_name")
    if isinstance(raw_sampler, str) and raw_sampler.strip():
        sampler_name = raw_sampler.strip()

    raw_resolution = data.get("resolution")
    if isinstance(raw_resolution, str) and raw_resolution.strip():
        resolution = raw_resolution.strip()

    return steps, cfg_scale, sampler_name, resolution


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
        sampler_name: str = _DEFAULT_SAMPLER_A1111,
        resolution: str = "512",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._sampler_name = sampler_name
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
            "sampler_name": self._sampler_name,
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


class DrawThingsClient:
    """Client for Draw Things' HTTP API server (Settings > API Server > Protocol: HTTP).

    Draw Things implements the same POST /sdapi/v1/txt2img path and
    {"images": [base64, ...]} response shape as AUTOMATIC1111, but reads its
    CFG/sampler parameters under different keys (guidance_scale/sampler
    instead of cfg_scale/sampler_name) — confirmed against a real Draw
    Things server via curl. A separate client (rather than reusing
    Automatic1111Client) keeps each server's request body honest instead of
    guessing which key names it'll accept.

    Uses whatever model is currently loaded in Draw Things — there's no
    per-request model selection, unlike OpenRouter/ComfyUI.
    """

    def __init__(
        self,
        base_url: str,
        *,
        steps: int = _DEFAULT_STEPS,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        sampler_name: str = _DEFAULT_SAMPLER_DRAW_THINGS,
        resolution: str = "512",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._sampler_name = sampler_name
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
            "sampler": self._sampler_name,
            "steps": self._steps,
            "guidance_scale": self._cfg_scale,
            "width": width,
            "height": height,
        }
        try:
            resp = requests.post(f"{self.base_url}/sdapi/v1/txt2img", json=body, timeout=self._timeout)
        except Exception as e:  # noqa: BLE001
            msg = f"Network error contacting Draw Things: {e}"
            raise LocalImageError(msg) from e

        if resp.status_code != 200:
            msg = f"Draw Things returned status {resp.status_code}: {resp.text[:300]}"
            raise LocalImageError(msg)

        try:
            images = resp.json()["images"]
        except (ValueError, KeyError) as e:
            msg = "Draw Things response missing image data"
            raise LocalImageError(msg) from e

        if not images:
            msg = "Draw Things returned no images"
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
    sampler_name: str = _DEFAULT_SAMPLER,
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
                "sampler_name": sampler_name,
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
        sampler_name: str = _DEFAULT_SAMPLER,
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
        self._sampler_name = sampler_name
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
            prompt,
            self._negative_prompt,
            self._checkpoint,
            width,
            height,
            self._steps,
            self._cfg_scale,
            seed,
            self._sampler_name,
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
        # whatever's loaded in the WebUI), so "auto" can't detect SDXL or a
        # distilled model here — falls back to the standard SD1.5-safe
        # defaults unless overridden in Settings (either the Resolution
        # dropdown, or the Advanced JSON field for steps/cfg/sampler too).
        resolution = _resolve_resolution(config.local_image_resolution, checkpoint=None)
        steps, cfg_scale, sampler_name, resolution = _apply_advanced_overrides(
            _DEFAULT_STEPS, _DEFAULT_CFG_SCALE, _DEFAULT_SAMPLER_A1111, resolution, config.local_image_advanced
        )
        return Automatic1111Client(
            config.local_image_url, steps=steps, cfg_scale=cfg_scale, sampler_name=sampler_name, resolution=resolution
        )

    if config.local_image_backend == "draw_things":
        # Same reasoning as the automatic1111 branch above — no checkpoint
        # info is available, so "auto" falls back to the standard defaults
        # unless overridden in Settings.
        resolution = _resolve_resolution(config.local_image_resolution, checkpoint=None)
        steps, cfg_scale, sampler_name, resolution = _apply_advanced_overrides(
            _DEFAULT_STEPS, _DEFAULT_CFG_SCALE, _DEFAULT_SAMPLER_DRAW_THINGS, resolution, config.local_image_advanced
        )
        return DrawThingsClient(
            config.local_image_url, steps=steps, cfg_scale=cfg_scale, sampler_name=sampler_name, resolution=resolution
        )

    if config.local_image_backend == "comfyui":
        checkpoint = config.local_image_checkpoint.strip()
        if not checkpoint:
            return None
        steps, cfg_scale, sampler_name = _pick_sampling_defaults(checkpoint)
        resolution = _resolve_resolution(config.local_image_resolution, checkpoint=checkpoint)
        steps, cfg_scale, sampler_name, resolution = _apply_advanced_overrides(
            steps, cfg_scale, sampler_name, resolution, config.local_image_advanced
        )
        return ComfyUIClient(
            config.local_image_url,
            checkpoint,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            resolution=resolution,
        )

    return None


_HEALTH_CHECK_PATHS = {
    "automatic1111": "/sdapi/v1/sd-models",
    "comfyui": "/system_stats",
    # Draw Things' HTTP API server doesn't implement AUTOMATIC1111's
    # sd-models/system_stats endpoints — just check the server answers at
    # the root at all.
    "draw_things": "/",
}


def validate_local_image_backend(backend: str, base_url: str) -> tuple[bool, str | None]:
    """Check that a local image generation backend is reachable.

    Args:
        backend: "automatic1111", "draw_things", or "comfyui".
        base_url: Base URL of the backend, e.g. 'http://127.0.0.1:7860'.

    Returns:
        Tuple (is_reachable, error_message). If reachable — (True, None).
    """
    if not base_url or not base_url.strip():
        return False, "Base URL is empty"

    url = base_url.strip().rstrip("/")
    health_path = _HEALTH_CHECK_PATHS.get(backend, "/system_stats")

    try:
        resp = requests.get(f"{url}{health_path}", timeout=10)
    except Exception as e:  # noqa: BLE001
        return False, f"Network error: {e}"

    if resp.status_code == 200:
        return True, None

    return False, f"Unexpected response: {resp.status_code}"
