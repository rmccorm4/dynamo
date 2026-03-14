# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.encode_utils — encoder
component extraction, image embedding encoding, and Qwen image feature
extraction with vLLM / AutoModel API selection."""

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

import dynamo.vllm.multimodal_utils.model as model_mod  # noqa: E402
from dynamo.vllm.multimodal_utils.encode_utils import (  # noqa: E402
    encode_image_embeddings,
    get_embedding_hash,
    get_encoder_components,
    get_qwen_image_features,
)
from dynamo.vllm.multimodal_utils.model import (  # noqa: E402
    ModelFamily,
    detect_model_family,
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


def _mock_vision_module(device="cpu", out_hidden_size=1024):
    """Create a mock nn.Module usable as a vision encoder."""
    mod = MagicMock(spec=torch.nn.Module)
    mod.device = torch.device(device)
    mod.out_hidden_size = out_hidden_size
    return mod


# ---------------------------------------------------------------------------
# get_embedding_hash
# ---------------------------------------------------------------------------


class TestGetEmbeddingHash:
    def test_deterministic(self):
        h1 = get_embedding_hash("https://example.com/img.jpg")
        h2 = get_embedding_hash("https://example.com/img.jpg")
        assert h1 == h2

    def test_different_keys_different_hashes(self):
        h1 = get_embedding_hash("a")
        h2 = get_embedding_hash("b")
        assert h1 != h2


# ---------------------------------------------------------------------------
# get_encoder_components
# ---------------------------------------------------------------------------


class TestGetEncoderComponents:
    def setup_method(self):
        detect_model_family.cache_clear()

    # -- Qwen VL, vLLM loaded -----------------------------------------------

    def test_qwen_vl_vllm_extracts_visual(self):
        visual_mod = _mock_vision_module()
        full_model = MagicMock(spec=torch.nn.Module)
        full_model.visual = visual_mod

        with (
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
                return_value=ModelFamily.QWEN_VL,
            ),
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
                return_value=True,
            ),
        ):
            encoder, projector = get_encoder_components(
                "Qwen/Qwen2-VL-2B", full_model
            )

        assert encoder is visual_mod
        assert projector is None

    def test_qwen_vl_vllm_missing_visual_falls_back(self):
        """If .visual is missing (unexpected), fall back to the model itself."""
        full_model = MagicMock(spec=torch.nn.Module)
        del full_model.visual

        with (
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
                return_value=ModelFamily.QWEN_VL,
            ),
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
                return_value=True,
            ),
        ):
            encoder, projector = get_encoder_components(
                "Qwen/Qwen2-VL-2B", full_model
            )

        assert encoder is full_model
        assert projector is None

    # -- Qwen VL, AutoModel loaded ------------------------------------------

    def test_qwen_vl_automodel_returns_full_model(self):
        full_model = MagicMock(spec=torch.nn.Module)

        with (
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
                return_value=ModelFamily.QWEN_VL,
            ),
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
                return_value=False,
            ),
        ):
            encoder, projector = get_encoder_components(
                "Qwen/Qwen2-VL-2B", full_model
            )

        assert encoder is full_model
        assert projector is None

    # -- LLaVA ---------------------------------------------------------------

    def test_llava_extracts_tower_and_projector(self):
        tower = _mock_vision_module()
        proj = MagicMock(spec=torch.nn.Module)
        full_model = MagicMock(spec=torch.nn.Module)
        full_model.vision_tower = tower
        full_model.multi_modal_projector = proj

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.LLAVA,
        ):
            encoder, projector = get_encoder_components(
                "llava-hf/llava-1.5-7b", full_model
            )

        assert encoder is tower
        assert projector is proj

    def test_llava_without_projector(self):
        tower = _mock_vision_module()
        full_model = MagicMock(spec=torch.nn.Module)
        full_model.vision_tower = tower
        full_model.multi_modal_projector = None

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.LLAVA,
        ):
            encoder, projector = get_encoder_components(
                "llava-hf/llava-1.5-7b", full_model
            )

        assert encoder is tower
        assert projector is None

    # -- Generic model with known attribute ----------------------------------

    def test_generic_probes_visual_attribute(self):
        visual_mod = MagicMock(spec=torch.nn.Module)
        full_model = MagicMock(spec=torch.nn.Module)
        del full_model.vision_tower
        full_model.visual = visual_mod

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            encoder, projector = get_encoder_components("org/some-vlm", full_model)

        assert encoder is visual_mod
        assert projector is None

    def test_generic_probes_vision_encoder_attribute(self):
        ve = MagicMock(spec=torch.nn.Module)
        full_model = MagicMock(spec=torch.nn.Module)
        del full_model.vision_tower
        del full_model.visual
        del full_model.vision_model
        full_model.vision_encoder = ve

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            encoder, projector = get_encoder_components("org/custom-vlm", full_model)

        assert encoder is ve
        assert projector is None

    # -- Generic model with no recognizable attribute ------------------------

    def test_generic_no_encoder_raises(self):
        full_model = MagicMock(spec=torch.nn.Module)
        del full_model.vision_tower
        del full_model.visual
        del full_model.vision_model
        del full_model.vision_encoder
        del full_model.image_encoder

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            with pytest.raises(
                NotImplementedError, match="Could not extract vision encoder"
            ):
                get_encoder_components("org/text-only-model", full_model)


# ---------------------------------------------------------------------------
# get_qwen_image_features
# ---------------------------------------------------------------------------


class TestGetQwenImageFeatures:
    def test_vllm_path_calls_forward_with_list_grid(self):
        encoder = _mock_vision_module()
        expected = torch.randn(256, 1024)
        encoder.return_value = expected

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 14, 14]]),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
            return_value=True,
        ):
            result = get_qwen_image_features(encoder, image_embeds)

        assert result is expected
        # Verify grid_thw was converted to list
        call_kwargs = encoder.call_args[1]
        assert isinstance(call_kwargs["grid_thw"], list)

    def test_automodel_path_calls_get_image_features(self):
        encoder = _mock_vision_module()
        expected = torch.randn(256, 1024)
        encoder.get_image_features = MagicMock(return_value=expected)

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 14, 14]]),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
            return_value=False,
        ):
            result = get_qwen_image_features(encoder, image_embeds)

        assert result is expected
        encoder.get_image_features.assert_called_once()

    def test_missing_grid_thw_raises(self):
        encoder = _mock_vision_module()
        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with pytest.raises(ValueError, match="grid_thw is not provided"):
            get_qwen_image_features(encoder, image_embeds)


# ---------------------------------------------------------------------------
# encode_image_embeddings
# ---------------------------------------------------------------------------


class TestEncodeImageEmbeddings:
    def setup_method(self):
        detect_model_family.cache_clear()

    def test_qwen_path(self):
        encoder = _mock_vision_module()
        raw_features = torch.randn(256, 1024)
        encoder.return_value = raw_features

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 14, 14]]),
        }

        with (
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
                return_value=ModelFamily.QWEN_VL,
            ),
            patch(
                "dynamo.vllm.multimodal_utils.encode_utils.is_vllm_encoder_active",
                return_value=True,
            ),
        ):
            result = encode_image_embeddings(
                "Qwen/Qwen2-VL-2B", image_embeds, encoder
            )

        # 2D → unsqueeze to 3D
        assert result.ndim == 3

    def test_llava_path_with_projector(self):
        encoder = _mock_vision_module()
        projector = MagicMock(spec=torch.nn.Module)

        vision_out = SimpleNamespace(last_hidden_state=torch.randn(1, 576, 1024))
        encoder.return_value = vision_out
        projector.return_value = torch.randn(1, 576, 4096)

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.LLAVA,
        ):
            result = encode_image_embeddings(
                "llava-hf/llava-1.5-7b", image_embeds, encoder, projector
            )

        assert result.shape == (1, 576, 4096)
        projector.assert_called_once()

    def test_generic_path_no_projector(self):
        encoder = _mock_vision_module()
        raw = torch.randn(1, 196, 768)
        encoder.return_value = raw

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            result = encode_image_embeddings(
                "org/generic-model", image_embeds, encoder
            )

        assert result.shape == (1, 196, 768)

    def test_generic_path_with_last_hidden_state(self):
        encoder = _mock_vision_module()
        vision_out = SimpleNamespace(last_hidden_state=torch.randn(1, 196, 768))
        encoder.return_value = vision_out

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            result = encode_image_embeddings(
                "org/generic-model", image_embeds, encoder
            )

        assert result.shape == (1, 196, 768)

    def test_generic_path_with_tuple_output(self):
        encoder = _mock_vision_module()
        t = torch.randn(1, 196, 768)
        encoder.return_value = (t, torch.randn(1, 196, 768))

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            result = encode_image_embeddings(
                "org/generic-model", image_embeds, encoder
            )

        assert result.shape == (1, 196, 768)

    def test_2d_output_gets_unsqueezed(self):
        encoder = _mock_vision_module()
        encoder.return_value = torch.randn(196, 768)

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.GENERIC,
        ):
            result = encode_image_embeddings(
                "org/model", image_embeds, encoder
            )

        assert result.ndim == 3
        assert result.shape == (1, 196, 768)

    def test_projector_with_direct_tensor_output(self):
        """When vision_encoder returns a raw tensor (no last_hidden_state),
        projector should still receive and process it."""
        encoder = _mock_vision_module()
        projector = MagicMock(spec=torch.nn.Module)

        raw = torch.randn(1, 576, 1024)
        encoder.return_value = raw
        projector.return_value = torch.randn(1, 576, 4096)

        image_embeds = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with patch(
            "dynamo.vllm.multimodal_utils.encode_utils.detect_model_family",
            return_value=ModelFamily.LLAVA,
        ):
            result = encode_image_embeddings(
                "llava-hf/llava-1.5-7b", image_embeds, encoder, projector
            )

        assert result.shape == (1, 576, 4096)
        # Projector was called with the raw tensor (not .last_hidden_state)
        projector.assert_called_once_with(raw)
