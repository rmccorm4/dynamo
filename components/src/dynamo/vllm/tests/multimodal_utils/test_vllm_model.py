# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.model — dynamic model family
detection, vision model loading with vLLM-first / AutoModel fallback, and
multimodal data construction helpers."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub heavy transitive dependencies that are unavailable in CI / CPU-only
# environments *before* importing the module under test.
# ---------------------------------------------------------------------------
_STUBS = {}
for _mod_name in (
    "vllm",
    "vllm.utils",
    "vllm.utils.system_utils",
    "transformers",
):
    if _mod_name not in sys.modules:
        _STUBS[_mod_name] = MagicMock()
        sys.modules[_mod_name] = _STUBS[_mod_name]

# Now we can import the module under test without pulling in vLLM / transformers
import dynamo.vllm.multimodal_utils.model as model_mod  # noqa: E402
from dynamo.vllm.multimodal_utils.model import (  # noqa: E402
    ModelFamily,
    _construct_qwen_image_data,
    construct_mm_data,
    construct_qwen_decode_mm_data,
    detect_model_family,
    is_qwen_vl_model,
    is_video_model,
    is_vllm_encoder_active,
    load_vision_model,
    normalize_model_name,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(model_type: str, architectures: list | None = None):
    """Build a minimal mock HuggingFace config object."""
    cfg = SimpleNamespace(model_type=model_type)
    if architectures is not None:
        cfg.architectures = architectures
    return cfg


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_simple_hf_name(self):
        assert (
            normalize_model_name("Qwen/Qwen2.5-VL-7B-Instruct")
            == "Qwen/Qwen2.5-VL-7B-Instruct"
        )

    def test_cache_path(self):
        path = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/abc123"
        assert normalize_model_name(path) == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_plain_string(self):
        assert normalize_model_name("some-model") == "some-model"


# ---------------------------------------------------------------------------
# detect_model_family  (AutoConfig is mocked)
# ---------------------------------------------------------------------------


class TestDetectModelFamily:
    """Tests for config-based model family detection."""

    def setup_method(self):
        detect_model_family.cache_clear()

    # -- model_type based detection -----------------------------------------

    @pytest.mark.parametrize(
        "model_type, expected",
        [
            ("qwen2_vl", ModelFamily.QWEN_VL),
            ("qwen2_5_vl", ModelFamily.QWEN_VL),
            ("qwen3_vl", ModelFamily.QWEN_VL),
            ("llava", ModelFamily.LLAVA),
            ("llava_next", ModelFamily.LLAVA),
            ("llava_next_video", ModelFamily.LLAVA_VIDEO),
        ],
    )
    def test_model_type_detection(self, model_type, expected):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config(model_type),
        ):
            assert detect_model_family("org/some-model") == expected

    def test_unknown_model_type_returns_generic(self):
        detect_model_family.cache_clear()
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("bert", architectures=["BertModel"]),
        ):
            assert detect_model_family("bert-base") == ModelFamily.GENERIC

    # -- architecture-based fallback ----------------------------------------

    def test_architecture_fallback_qwen_vl(self):
        cfg = _make_config(
            "unknown", architectures=["Qwen2VLForConditionalGeneration"]
        )
        with patch.object(
            model_mod.AutoConfig, "from_pretrained", return_value=cfg
        ):
            assert detect_model_family("org/custom-qwen-vl") == ModelFamily.QWEN_VL

    def test_architecture_fallback_llava(self):
        cfg = _make_config(
            "unknown", architectures=["LlavaForConditionalGeneration"]
        )
        with patch.object(
            model_mod.AutoConfig, "from_pretrained", return_value=cfg
        ):
            assert detect_model_family("org/custom-llava") == ModelFamily.LLAVA

    def test_architecture_fallback_llava_video(self):
        cfg = _make_config(
            "unknown",
            architectures=["LlavaNextVideoForConditionalGeneration"],
        )
        with patch.object(
            model_mod.AutoConfig, "from_pretrained", return_value=cfg
        ):
            assert detect_model_family("org/custom-video") == ModelFamily.LLAVA_VIDEO

    # -- error handling -----------------------------------------------------

    def test_config_load_failure_returns_generic(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            side_effect=OSError("model not found"),
        ):
            assert detect_model_family("nonexistent/model") == ModelFamily.GENERIC

    # -- caching ------------------------------------------------------------

    def test_results_are_cached(self):
        detect_model_family.cache_clear()
        mock_auto = MagicMock(return_value=_make_config("qwen2_vl"))
        with patch.object(model_mod.AutoConfig, "from_pretrained", mock_auto):
            result1 = detect_model_family("Qwen/Qwen2-VL-2B")
            result2 = detect_model_family("Qwen/Qwen2-VL-2B")

        assert result1 is result2 is ModelFamily.QWEN_VL
        # AutoConfig should only be called once thanks to caching
        mock_auto.assert_called_once()


# ---------------------------------------------------------------------------
# is_qwen_vl_model / is_video_model  (thin wrappers)
# ---------------------------------------------------------------------------


class TestModelFamilyHelpers:
    def setup_method(self):
        detect_model_family.cache_clear()

    def test_is_qwen_vl_model_true(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("qwen2_5_vl"),
        ):
            assert is_qwen_vl_model("Qwen/Qwen2.5-VL-7B") is True

    def test_is_qwen_vl_model_false(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava"),
        ):
            assert is_qwen_vl_model("llava-hf/llava-1.5-7b") is False

    def test_is_video_model_true(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava_next_video"),
        ):
            assert is_video_model("llava-hf/LLaVA-NeXT-Video-7B") is True

    def test_is_video_model_false(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("qwen2_vl"),
        ):
            assert is_video_model("Qwen/Qwen2-VL-2B") is False


# ---------------------------------------------------------------------------
# load_vision_model
# ---------------------------------------------------------------------------


class TestLoadVisionModel:
    """Tests for vLLM-first loading with AutoModel fallback."""

    def setup_method(self):
        detect_model_family.cache_clear()

    def _build_vllm_model_chain(self):
        """Build a mock vLLM LLM object with the nested attribute chain that
        ``load_vision_model`` traverses to extract the model runner model."""
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.visual = MagicMock(spec=torch.nn.Module)

        vllm_obj = MagicMock()
        (
            vllm_obj.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
        ) = mock_model
        return vllm_obj, mock_model

    @patch.object(model_mod, "update_environment_variables")
    @patch.object(model_mod, "LLM")
    def test_vllm_success_sets_active_flag(self, mock_llm_cls, mock_env):
        vllm_obj, expected_model = self._build_vllm_model_chain()
        mock_llm_cls.return_value = vllm_obj

        result = load_vision_model("Qwen/Qwen2-VL-2B")

        assert result is expected_model
        assert is_vllm_encoder_active() is True
        mock_llm_cls.assert_called_once()
        # Verify mm_encoder_only=True is passed
        _, kwargs = mock_llm_cls.call_args
        assert kwargs["mm_encoder_only"] is True

    @patch.object(model_mod, "AutoModel")
    @patch.object(model_mod, "update_environment_variables")
    @patch.object(
        model_mod,
        "LLM",
        side_effect=RuntimeError("encoder-only not supported"),
    )
    def test_vllm_failure_falls_back_to_automodel(
        self, mock_llm_cls, mock_env, mock_auto
    ):
        fallback_model = MagicMock(spec=torch.nn.Module)
        mock_auto.from_pretrained.return_value = fallback_model

        result = load_vision_model("org/unsupported-model")

        assert result is fallback_model
        assert is_vllm_encoder_active() is False
        mock_auto.from_pretrained.assert_called_once_with(
            "org/unsupported-model",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    @patch.object(model_mod, "AutoModel")
    def test_vllm_encoder_env_var_zero_skips_vllm(self, mock_auto):
        fallback_model = MagicMock(spec=torch.nn.Module)
        mock_auto.from_pretrained.return_value = fallback_model

        original = model_mod.VLLM_ENCODER
        try:
            model_mod.VLLM_ENCODER = 0
            result = load_vision_model("any/model")
            assert result is fallback_model
            assert is_vllm_encoder_active() is False
        finally:
            model_mod.VLLM_ENCODER = original


# ---------------------------------------------------------------------------
# construct_mm_data
# ---------------------------------------------------------------------------


class TestConstructMmData:
    def setup_method(self):
        detect_model_family.cache_clear()

    def test_video_model(self):
        import numpy as np

        video = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava_next_video"),
        ):
            result = construct_mm_data(
                "llava-hf/LLaVA-NeXT-Video-7B",
                torch.float16,
                video_numpy=video,
            )
        assert "video" in result

    def test_video_model_missing_frames_raises(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava_next_video"),
        ):
            with pytest.raises(ValueError, match="No video frames"):
                construct_mm_data(
                    "llava-hf/LLaVA-NeXT-Video-7B",
                    torch.float16,
                )

    def test_qwen_vl_model_builds_dict(self):
        embeds = torch.randn(1, 256, 1024)
        grid = [[1, 16, 16]]
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("qwen2_vl"),
        ):
            result = construct_mm_data(
                "Qwen/Qwen2-VL-2B",
                torch.float16,
                image_embeds=embeds,
                image_grid_thw=grid,
            )
        assert isinstance(result["image"], dict)
        assert "image_embeds" in result["image"]
        assert "image_grid_thw" in result["image"]

    def test_llava_model_returns_tensor(self):
        embeds = torch.randn(1, 576, 4096)
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava"),
        ):
            result = construct_mm_data(
                "llava-hf/llava-1.5-7b",
                torch.float16,
                image_embeds=embeds,
            )
        assert isinstance(result["image"], torch.Tensor)

    def test_generic_model_with_grid_thw_uses_qwen_style(self):
        """Models that provide grid info but aren't recognized as Qwen should
        still get the dict-style mm_data (e.g. future Qwen variants)."""
        embeds = torch.randn(1, 256, 1024)
        grid = [[1, 16, 16]]
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("some_new_vl"),
        ):
            result = construct_mm_data(
                "org/future-vl-model",
                torch.float16,
                image_embeds=embeds,
                image_grid_thw=grid,
            )
        assert isinstance(result["image"], dict)

    def test_missing_image_embeds_raises(self):
        with patch.object(
            model_mod.AutoConfig,
            "from_pretrained",
            return_value=_make_config("llava"),
        ):
            with pytest.raises(ValueError, match="No image embeddings"):
                construct_mm_data(
                    "llava-hf/llava-1.5-7b",
                    torch.float16,
                )


# ---------------------------------------------------------------------------
# _construct_qwen_image_data
# ---------------------------------------------------------------------------


class TestConstructQwenImageData:
    def test_valid_input(self):
        embeds = torch.randn(1, 256, 1024)
        grid = [[1, 16, 16]]
        result = _construct_qwen_image_data(embeds, grid)
        assert "image" in result
        assert result["image"]["image_embeds"].shape == (256, 1024)
        assert result["image"]["image_grid_thw"].tolist() == [[1, 16, 16]]

    def test_missing_grid_raises(self):
        embeds = torch.randn(1, 256, 1024)
        with pytest.raises(ValueError, match="No image grid"):
            _construct_qwen_image_data(embeds, None)

    def test_empty_grid_raises(self):
        embeds = torch.randn(1, 256, 1024)
        with pytest.raises(ValueError, match="No image grid"):
            _construct_qwen_image_data(embeds, [])


# ---------------------------------------------------------------------------
# construct_qwen_decode_mm_data  (unchanged, but verify still works)
# ---------------------------------------------------------------------------


class TestConstructQwenDecodeMmData:
    def test_basic_construction(self):
        mm_data = construct_qwen_decode_mm_data(
            image_grid_thw=[16, 16],
            embeddings_shape=[2, 1024],
            request_id="test-req-1",
        )
        assert "image" in mm_data
        assert "image_grid_thw" in mm_data["image"]
        assert "image_embeds" in mm_data["image"]
        assert mm_data["image"]["image_embeds"].shape == (2, 1024)
        assert torch.allclose(
            mm_data["image"]["image_grid_thw"], torch.tensor([16, 16])
        )

    def test_counter_wraps_around(self):
        """Verify the counter wraps without error after exceeding dtype max."""
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        for i in range(max_rounds):
            try:
                mm_data = construct_qwen_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_qwen_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
            assert mm_data["image"]["image_embeds"].shape == (2, 1024)

    def test_missing_grid_raises(self):
        with pytest.raises(ValueError, match="No image grid"):
            construct_qwen_decode_mm_data(
                image_grid_thw=None,
                embeddings_shape=[2, 1024],
                request_id="x",
            )

    def test_missing_embeddings_shape_raises(self):
        with pytest.raises(ValueError, match="embeddings_shape is required"):
            construct_qwen_decode_mm_data(
                image_grid_thw=[16, 16],
                embeddings_shape=None,
                request_id="x",
            )
