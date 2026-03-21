"""Reading, writing, and validation of AnkiForge configuration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from ankiforge.models import AddonConfig, ModelPricingCache

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]

_OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"


def _detect_addon_module() -> str:
    """Detect addon folder name by walking up until we find the addons21 parent."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.parent.name == "addons21":
            return parent.name
    return "ankiforge"


_ADDON_MODULE = _detect_addon_module()


def _get_mw() -> AnkiQt:
    """Get the Anki main window (mw).

    Raises:
        RuntimeError: If Anki runtime is not available.
    """
    try:
        from aqt import mw  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "Anki runtime is not available"
        raise RuntimeError(msg) from e

    if mw is None:
        msg = "Anki main window is not initialized"
        raise RuntimeError(msg)

    return mw


def _fallback_config_path() -> Path:
    """Path to config.json in the package (for fallback outside Anki)."""
    return Path(__file__).parent.parent / "config.json"


def _parse_pricing(data: object) -> ModelPricingCache | None:
    """Parse pricing from a config dictionary."""
    if not isinstance(data, dict):
        return None
    return ModelPricingCache(
        prompt=float(data.get("prompt", 0) or 0),
        completion=float(data.get("completion", 0) or 0),
        image=float(data.get("image", 0) or 0),
        request=float(data.get("request", 0) or 0),
    )


def _dict_to_config(data: dict[str, object] | None) -> AddonConfig:
    """Convert a dictionary to AddonConfig with defaults for missing fields."""
    if not data:
        return AddonConfig()

    defaults = AddonConfig()
    cached_balance_raw = data.get("cached_balance")
    cached_balance = (
        float(cached_balance_raw)  # type: ignore[arg-type]
        if cached_balance_raw is not None
        else None
    )
    return AddonConfig(
        api_key=str(data.get("api_key", defaults.api_key)),
        text_model=str(data.get("text_model", defaults.text_model)),
        image_model=str(data.get("image_model", defaults.image_model)),
        audio_model=str(data.get("audio_model", defaults.audio_model)),
        language=str(data.get("language", defaults.language)),
        text_model_pricing=_parse_pricing(data.get("text_model_pricing")),
        image_model_pricing=_parse_pricing(data.get("image_model_pricing")),
        audio_model_pricing=_parse_pricing(data.get("audio_model_pricing")),
        cached_balance=cached_balance,
    )


def get_config() -> AddonConfig:
    """Read add-on configuration.

    First tries to read via Anki Config API.
    If Anki is unavailable — reads from config.json in the package.

    Returns:
        Current configuration.
    """
    try:
        mw = _get_mw()
        data = mw.addonManager.getConfig(_ADDON_MODULE)
        return _dict_to_config(data)
    except RuntimeError:
        pass

    # Fallback: read config.json from the package
    config_path = _fallback_config_path()
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return _dict_to_config(data)

    return AddonConfig()


def save_config(config: AddonConfig) -> None:
    """Save add-on configuration.

    First tries to write via Anki Config API.
    If Anki is unavailable — writes to config.json.

    Args:
        config: Configuration to save.
    """
    data = asdict(config)

    try:
        mw = _get_mw()
        mw.addonManager.writeConfig(_ADDON_MODULE, data)
        return
    except RuntimeError:
        pass

    # Fallback: write to JSON file
    config_path = _fallback_config_path()
    config_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def is_configured() -> bool:
    """Check whether the add-on is configured (API key is set).

    Returns:
        True if API key is present.
    """
    try:
        config = get_config()
        return bool(config.api_key)
    except Exception:  # noqa: BLE001
        return False


def validate_api_key(api_key: str) -> tuple[bool, str | None]:
    """Validate an OpenRouter API key.

    Checks the format and makes a request to the OpenRouter API.

    Args:
        api_key: API key to validate.

    Returns:
        Tuple (is_valid, error_message). If valid — (True, None).
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"

    if not api_key.startswith("sk-or-"):
        return False, "API key must start with 'sk-or-'"

    # Verify key with a request to OpenRouter
    try:
        resp = requests.get(
            _OPENROUTER_AUTH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"Network error: {e}"

    if resp.status_code == 200:
        return True, None

    if resp.status_code == 401:
        return False, "Invalid API key"

    return False, f"Unexpected response: {resp.status_code}"
