"""Tests for local image generation backends (Automatic1111, ComfyUI)."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from ankiforge.local_image.client import (
    _DEFAULT_CFG_SCALE,
    _DEFAULT_STEPS,
    _DISTILLED_CFG_SCALE,
    _DISTILLED_STEPS,
    _SIZE_TIERS,
    Automatic1111Client,
    ComfyUIClient,
    LocalImageError,
    _looks_sdxl_checkpoint,
    _pick_sampling_defaults,
    _resolve_resolution,
    _resolve_size,
    build_local_image_client,
    validate_local_image_backend,
)
from ankiforge.models import AddonConfig


class TestAutomatic1111Client:
    def test_generate_image_returns_decoded_bytes(self) -> None:
        raw_png = b"fake-png-bytes"
        encoded = base64.b64encode(raw_png).decode()
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"images": [encoded]}

        with patch("ankiforge.local_image.client.requests.post", return_value=mock_response) as mock_post:
            client = Automatic1111Client("http://127.0.0.1:7860")
            result = client.generate_image("a cat", size="square")

        assert result == raw_png
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:7860/sdapi/v1/txt2img"
        assert kwargs["json"]["prompt"] == "a cat"
        assert kwargs["json"]["width"] == 512
        assert kwargs["json"]["height"] == 512

    def test_strips_trailing_slash_from_base_url(self) -> None:
        client = Automatic1111Client("http://127.0.0.1:7860/")
        assert client.base_url == "http://127.0.0.1:7860"

    def test_non_200_response_raises(self) -> None:
        mock_response = MagicMock(status_code=500, text="server error")

        with patch("ankiforge.local_image.client.requests.post", return_value=mock_response):
            client = Automatic1111Client("http://127.0.0.1:7860")
            with pytest.raises(LocalImageError, match="status 500"):
                client.generate_image("a cat")

    def test_missing_images_field_raises(self) -> None:
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {}

        with patch("ankiforge.local_image.client.requests.post", return_value=mock_response):
            client = Automatic1111Client("http://127.0.0.1:7860")
            with pytest.raises(LocalImageError, match="missing image data"):
                client.generate_image("a cat")

    def test_empty_images_list_raises(self) -> None:
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"images": []}

        with patch("ankiforge.local_image.client.requests.post", return_value=mock_response):
            client = Automatic1111Client("http://127.0.0.1:7860")
            with pytest.raises(LocalImageError, match="no images"):
                client.generate_image("a cat")

    def test_network_error_raises(self) -> None:
        with patch("ankiforge.local_image.client.requests.post", side_effect=Exception("refused")):
            client = Automatic1111Client("http://127.0.0.1:7860")
            with pytest.raises(LocalImageError, match="Network error"):
                client.generate_image("a cat")


class TestComfyUIClient:
    def test_requires_checkpoint(self) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            ComfyUIClient("http://127.0.0.1:8188", "")

    def test_generate_image_submits_and_polls_and_fetches(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"prompt_id": "abc123"}

        history_resp = MagicMock(status_code=200)
        history_resp.json.return_value = {
            "abc123": {
                "outputs": {
                    "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]},
                }
            }
        }

        view_resp = MagicMock(status_code=200, content=b"image-bytes")

        with (
            patch("ankiforge.local_image.client.requests.post", return_value=submit_resp) as mock_post,
            patch(
                "ankiforge.local_image.client.requests.get",
                side_effect=[history_resp, view_resp],
            ) as mock_get,
        ):
            client = ComfyUIClient("http://127.0.0.1:8188", "sd_xl_turbo.safetensors", poll_interval=0)
            result = client.generate_image("a dog", size="landscape")

        assert result == b"image-bytes"
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == "http://127.0.0.1:8188/prompt"
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0].args[0] == "http://127.0.0.1:8188/history/abc123"
        assert mock_get.call_args_list[1].args[0] == "http://127.0.0.1:8188/view"

    def test_missing_prompt_id_raises(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {}

        with patch("ankiforge.local_image.client.requests.post", return_value=submit_resp):
            client = ComfyUIClient("http://127.0.0.1:8188", "ckpt.safetensors")
            with pytest.raises(LocalImageError, match="prompt_id"):
                client.generate_image("a dog")

    def test_non_200_submit_raises(self) -> None:
        submit_resp = MagicMock(status_code=400, text="bad workflow")

        with patch("ankiforge.local_image.client.requests.post", return_value=submit_resp):
            client = ComfyUIClient("http://127.0.0.1:8188", "ckpt.safetensors")
            with pytest.raises(LocalImageError, match="status 400"):
                client.generate_image("a dog")

    def test_timeout_waiting_for_completion_raises(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"prompt_id": "abc123"}
        pending_resp = MagicMock(status_code=200)
        pending_resp.json.return_value = {}

        with (
            patch("ankiforge.local_image.client.requests.post", return_value=submit_resp),
            patch("ankiforge.local_image.client.requests.get", return_value=pending_resp),
        ):
            client = ComfyUIClient("http://127.0.0.1:8188", "ckpt.safetensors", timeout=0, poll_interval=0)
            with pytest.raises(LocalImageError, match="Timed out"):
                client.generate_image("a dog")

    def test_no_images_in_outputs_raises(self) -> None:
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"prompt_id": "abc123"}
        history_resp = MagicMock(status_code=200)
        history_resp.json.return_value = {"abc123": {"outputs": {"9": {}}}}

        with (
            patch("ankiforge.local_image.client.requests.post", return_value=submit_resp),
            patch("ankiforge.local_image.client.requests.get", return_value=history_resp),
        ):
            client = ComfyUIClient("http://127.0.0.1:8188", "ckpt.safetensors", poll_interval=0)
            with pytest.raises(LocalImageError, match="no images"):
                client.generate_image("a dog")


class TestBuildLocalImageClient:
    def test_returns_none_when_no_backend(self) -> None:
        assert build_local_image_client(AddonConfig()) is None

    def test_returns_none_when_backend_set_but_no_url(self) -> None:
        config = AddonConfig(local_image_backend="automatic1111")
        assert build_local_image_client(config) is None

    def test_builds_automatic1111_client(self) -> None:
        config = AddonConfig(local_image_backend="automatic1111", local_image_url="http://127.0.0.1:7860")
        client = build_local_image_client(config)
        assert isinstance(client, Automatic1111Client)
        assert client.base_url == "http://127.0.0.1:7860"

    def test_builds_comfyui_client(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="sd_xl_turbo.safetensors",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)

    def test_comfyui_without_checkpoint_returns_none(self) -> None:
        config = AddonConfig(local_image_backend="comfyui", local_image_url="http://127.0.0.1:8188")
        assert build_local_image_client(config) is None

    def test_unknown_backend_returns_none(self) -> None:
        config = AddonConfig(local_image_backend="bogus", local_image_url="http://127.0.0.1:9999")
        assert build_local_image_client(config) is None

    def test_automatic1111_uses_standard_defaults(self) -> None:
        """Regression test: Automatic1111 has no checkpoint info to detect a
        distilled model from, so it must use the safe standard defaults —
        not the old always-4-steps default, which produced noisy garbage on
        a normal (non-distilled) checkpoint."""
        config = AddonConfig(local_image_backend="automatic1111", local_image_url="http://127.0.0.1:7860")
        client = build_local_image_client(config)
        assert isinstance(client, Automatic1111Client)
        assert client._steps == _DEFAULT_STEPS  # noqa: SLF001
        assert client._cfg_scale == _DEFAULT_CFG_SCALE  # noqa: SLF001

    def test_comfyui_turbo_checkpoint_uses_distilled_defaults(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="sd_xl_turbo_1.0.safetensors",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)
        assert client._steps == _DISTILLED_STEPS  # noqa: SLF001
        assert client._cfg_scale == _DISTILLED_CFG_SCALE  # noqa: SLF001

    def test_comfyui_normal_checkpoint_uses_standard_defaults(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="realisticVisionV60B1.safetensors",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)
        assert client._steps == _DEFAULT_STEPS  # noqa: SLF001
        assert client._cfg_scale == _DEFAULT_CFG_SCALE  # noqa: SLF001


class TestPickSamplingDefaults:
    @pytest.mark.parametrize(
        "checkpoint",
        ["sd_xl_turbo_1.0.safetensors", "SDXL-Lightning-4step.safetensors", "dreamshaper_lcm.safetensors"],
    )
    def test_distilled_markers_pick_low_step_defaults(self, checkpoint: str) -> None:
        assert _pick_sampling_defaults(checkpoint) == (_DISTILLED_STEPS, _DISTILLED_CFG_SCALE)

    def test_normal_checkpoint_picks_standard_defaults(self) -> None:
        assert _pick_sampling_defaults("realisticVisionV60B1.safetensors") == (_DEFAULT_STEPS, _DEFAULT_CFG_SCALE)


class TestValidateLocalImageBackend:
    def test_empty_url_returns_false(self) -> None:
        is_valid, error = validate_local_image_backend("automatic1111", "")
        assert is_valid is False
        assert error is not None

    def test_automatic1111_reachable(self) -> None:
        mock_response = MagicMock(status_code=200)
        with patch("ankiforge.local_image.client.requests.get", return_value=mock_response) as mock_get:
            is_valid, error = validate_local_image_backend("automatic1111", "http://127.0.0.1:7860/")

        assert is_valid is True
        assert error is None
        mock_get.assert_called_once_with("http://127.0.0.1:7860/sdapi/v1/sd-models", timeout=10)

    def test_comfyui_reachable(self) -> None:
        mock_response = MagicMock(status_code=200)
        with patch("ankiforge.local_image.client.requests.get", return_value=mock_response) as mock_get:
            is_valid, error = validate_local_image_backend("comfyui", "http://127.0.0.1:8188")

        assert is_valid is True
        mock_get.assert_called_once_with("http://127.0.0.1:8188/system_stats", timeout=10)

    def test_unreachable_returns_false(self) -> None:
        mock_response = MagicMock(status_code=404)
        with patch("ankiforge.local_image.client.requests.get", return_value=mock_response):
            is_valid, error = validate_local_image_backend("automatic1111", "http://127.0.0.1:7860")

        assert is_valid is False
        assert error is not None

    def test_network_error_returns_false(self) -> None:
        with patch("ankiforge.local_image.client.requests.get", side_effect=Exception("refused")):
            is_valid, error = validate_local_image_backend("comfyui", "http://127.0.0.1:8188")

        assert is_valid is False
        assert error is not None


# ---------------------------------------------------------------------------
# Resolution: _looks_sdxl_checkpoint / _resolve_resolution / _resolve_size
# ---------------------------------------------------------------------------


class TestLooksSdxlCheckpoint:
    @pytest.mark.parametrize(
        "checkpoint",
        [
            "sd_xl_turbo_1.0.safetensors",
            "sdxl_base_1.0.safetensors",
            "juggernaut-xl-v9.safetensors",
            "SDXL.safetensors",
        ],
    )
    def test_matches_sdxl_family_names(self, checkpoint: str) -> None:
        assert _looks_sdxl_checkpoint(checkpoint) is True

    @pytest.mark.parametrize(
        "checkpoint",
        ["realisticVisionV60B1.safetensors", "dreamshaper_8.safetensors", "sd15_pruned.safetensors"],
    )
    def test_does_not_match_sd15_family_names(self, checkpoint: str) -> None:
        assert _looks_sdxl_checkpoint(checkpoint) is False


class TestResolveResolution:
    def test_explicit_setting_wins_over_detection(self) -> None:
        assert _resolve_resolution("768", checkpoint="sdxl_base.safetensors") == "768"

    def test_auto_detects_sdxl_checkpoint(self) -> None:
        assert _resolve_resolution("auto", checkpoint="sd_xl_turbo.safetensors") == "1024"

    def test_auto_falls_back_to_512_for_normal_checkpoint(self) -> None:
        assert _resolve_resolution("auto", checkpoint="dreamshaper_8.safetensors") == "512"

    def test_auto_falls_back_to_512_without_checkpoint_info(self) -> None:
        # Automatic1111 case — no checkpoint filename to detect from.
        assert _resolve_resolution("auto", checkpoint=None) == "512"

    def test_unknown_setting_falls_back_to_512(self) -> None:
        assert _resolve_resolution("bogus", checkpoint=None) == "512"


class TestResolveSize:
    def test_default_tier_matches_previous_512_behavior(self) -> None:
        assert _resolve_size("landscape") == (768, 512)

    def test_1024_tier(self) -> None:
        assert _resolve_size("square", "1024") == (1024, 1024)
        assert _resolve_size(None, "1024") == _SIZE_TIERS["1024"]["auto"]

    def test_unknown_tier_falls_back_to_512(self) -> None:
        assert _resolve_size("square", "bogus") == _SIZE_TIERS["512"]["square"]


class TestBuildLocalImageClientResolution:
    def test_comfyui_sdxl_checkpoint_gets_1024_resolution(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="sdxl_base_1.0.safetensors",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)
        assert client._resolution == "1024"  # noqa: SLF001

    def test_comfyui_normal_checkpoint_gets_512_resolution(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="dreamshaper_8.safetensors",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)
        assert client._resolution == "512"  # noqa: SLF001

    def test_comfyui_explicit_resolution_overrides_detection(self) -> None:
        config = AddonConfig(
            local_image_backend="comfyui",
            local_image_url="http://127.0.0.1:8188",
            local_image_checkpoint="dreamshaper_8.safetensors",
            local_image_resolution="1024",
        )
        client = build_local_image_client(config)
        assert isinstance(client, ComfyUIClient)
        assert client._resolution == "1024"  # noqa: SLF001

    def test_automatic1111_defaults_to_512(self) -> None:
        config = AddonConfig(local_image_backend="automatic1111", local_image_url="http://127.0.0.1:7860")
        client = build_local_image_client(config)
        assert isinstance(client, Automatic1111Client)
        assert client._resolution == "512"  # noqa: SLF001

    def test_automatic1111_explicit_resolution_is_respected(self) -> None:
        config = AddonConfig(
            local_image_backend="automatic1111",
            local_image_url="http://127.0.0.1:7860",
            local_image_resolution="1024",
        )
        client = build_local_image_client(config)
        assert isinstance(client, Automatic1111Client)
        assert client._resolution == "1024"  # noqa: SLF001
