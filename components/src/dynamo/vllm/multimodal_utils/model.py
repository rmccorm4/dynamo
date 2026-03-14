# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel
from vllm import LLM
from vllm.utils.system_utils import update_environment_variables

logger = logging.getLogger(__name__)

# Set VLLM_ENCODER=0 to skip the vLLM encoder attempt and always use AutoModel.
VLLM_ENCODER = int(os.getenv("VLLM_ENCODER", 1))


class ModelFamily(enum.Enum):
    """Detected vision model family, used for model-specific encoding behavior."""

    QWEN_VL = "qwen_vl"  # Qwen2-VL, Qwen2.5-VL, Qwen3-VL (uses image_grid_thw / mRoPE)
    LLAVA = "llava"  # LLaVA 1.5 and similar (vision_tower + projector)
    LLAVA_VIDEO = "llava_video"  # LLaVA-NeXT-Video
    GENERIC = "generic"  # Unknown model family


# HuggingFace model_type values for each family
_QWEN_VL_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl"})
_LLAVA_MODEL_TYPES = frozenset({"llava", "llava_next"})
_LLAVA_VIDEO_MODEL_TYPES = frozenset({"llava_next_video"})

# Tracks whether the last load_vision_model call succeeded via vLLM
_vllm_encoder_active = False


def normalize_model_name(model_name: str) -> str:
    """
    Extract and normalize model name from various formats including HuggingFace cache paths.

    Args:
        model_name: Model identifier which can be:
            - A simple model name: "Qwen/Qwen2.5-VL-7B-Instruct"
            - A HuggingFace cache path: "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/..."
            - A local path to a model directory

    Returns:
        Normalized model name in the format "organization/model-name"

    Examples:
        >>> normalize_model_name("Qwen/Qwen2.5-VL-7B-Instruct")
        "Qwen/Qwen2.5-VL-7B-Instruct"
        >>> normalize_model_name("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/...")
        "Qwen/Qwen2.5-VL-7B-Instruct"
    """
    # If it's already a simple model name (org/model format), return as-is
    if "/" in model_name and not model_name.startswith("/"):
        return model_name

    # Handle HuggingFace cache paths
    if "models--" in model_name:
        # Extract from cache path format: models--ORG--MODEL-NAME
        # Split on "models--" then on "--" to handle dashes in org/model names
        parts_after_models = model_name.split("models--", 1)
        if len(parts_after_models) > 1:
            # Split the remaining part on "--" and take the last two segments
            segments = parts_after_models[1].split("--")
            if len(segments) >= 2:
                # Take all segments except the last as org (rejoined with dashes)
                # and the last segment (before any slash) as model name
                org_segments = segments[:-1]
                model_segment = segments[-1].split("/")[
                    0
                ]  # Remove any path after model name

                org = "--".join(org_segments)  # Rejoin org parts with dashes
                model = model_segment
                return f"{org}/{model}"

    # Handle local directory paths - extract the last directory name
    path = Path(model_name)
    if path.exists() and path.is_dir():
        return path.name

    # If no pattern matches, return the original name
    return model_name


@lru_cache(maxsize=16)
def detect_model_family(model_name: str) -> ModelFamily:
    """Detect vision model family from HuggingFace config metadata.

    Uses the model's ``model_type`` and ``architectures`` fields to determine
    the model family without loading weights. Results are cached per model_name.
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()

        if model_type in _QWEN_VL_MODEL_TYPES:
            return ModelFamily.QWEN_VL
        if model_type in _LLAVA_VIDEO_MODEL_TYPES:
            return ModelFamily.LLAVA_VIDEO
        if model_type in _LLAVA_MODEL_TYPES:
            return ModelFamily.LLAVA

        # Fallback: check architectures for broader matching
        architectures = getattr(config, "architectures", []) or []
        arch_str = " ".join(architectures).lower()
        if "qwen" in arch_str and "vl" in arch_str:
            return ModelFamily.QWEN_VL
        if "video" in arch_str and "llava" in arch_str:
            return ModelFamily.LLAVA_VIDEO
        if "llava" in arch_str:
            return ModelFamily.LLAVA

        logger.info(
            "Model '%s' (model_type='%s', architectures=%s) not recognized as a known "
            "VLM family; treating as generic.",
            model_name,
            model_type,
            architectures,
        )
        return ModelFamily.GENERIC
    except Exception as e:
        logger.warning(
            "Could not load config for model '%s' to detect family: %s. "
            "Treating as generic.",
            model_name,
            e,
        )
        return ModelFamily.GENERIC


def is_qwen_vl_model(model_name: str) -> bool:
    """
    Check if a model is a Qwen VL variant using config-based detection.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is a Qwen VL variant, False otherwise
    """
    return detect_model_family(model_name) == ModelFamily.QWEN_VL


def is_video_model(model_name: str) -> bool:
    """
    Check if a model is a video model using config-based detection.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is a video model, False otherwise
    """
    return detect_model_family(model_name) == ModelFamily.LLAVA_VIDEO


def is_vllm_encoder_active() -> bool:
    """Return whether the vision model was successfully loaded via the vLLM encoder."""
    return _vllm_encoder_active


def load_vision_model(model_id: str, enforce_eager: bool = False) -> torch.nn.Module:
    """Load a vision model, trying vLLM's encoder-only mode first.

    Always attempts to load via vLLM's ``mm_encoder_only`` mode, which avoids
    loading full LLM weights and uses significantly less GPU memory. Falls back
    to ``AutoModel.from_pretrained()`` with a warning if vLLM fails (e.g. when
    the model architecture is not yet supported by vLLM's encoder-only path).

    Set ``VLLM_ENCODER=0`` to skip the vLLM attempt entirely.
    """
    global _vllm_encoder_active

    if not VLLM_ENCODER:
        logger.info(
            "VLLM_ENCODER=0: skipping vLLM encoder, using AutoModel for '%s'",
            model_id,
        )
        _vllm_encoder_active = False
        return AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    try:
        # Disable multiprocessing to get ViT from the same process
        update_environment_variables({"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})

        # Load only the vision model via vLLM to avoid loading full LLM weights.
        # Uses native vLLM encoder-only loading added in
        # https://github.com/vllm-project/vllm/pull/32605
        vllm_model = LLM(
            model=model_id,
            enforce_eager=enforce_eager,
            kv_cache_memory_bytes=64
            * 1024
            * 1024,  # 64MB: encoder-only doesn't need KV cache
            max_model_len=1,
            mm_encoder_only=True,
            enable_prefix_caching=False,
        )
        model = (
            vllm_model.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
        )
        _vllm_encoder_active = True
        logger.info("Loaded vision model via vLLM encoder for '%s'", model_id)
        return model
    except Exception as e:
        _vllm_encoder_active = False
        logger.warning(
            "vLLM encoder-only loading failed for '%s': %s. "
            "Falling back to AutoModel.from_pretrained(). "
            "This may use more GPU memory as the full model weights are loaded.",
            model_id,
            e,
        )
        return AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )


def construct_mm_data(
    model: str,
    embeddings_dtype: torch.dtype,
    image_embeds: Optional[torch.Tensor] = None,
    video_numpy: Optional[Any] = None,
    image_grid_thw: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""

    # Handle video models
    if is_video_model(model):
        if video_numpy is None:
            raise ValueError("No video frames provided.")
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    # Model-specific image handling
    if is_qwen_vl_model(model):
        return _construct_qwen_image_data(image_embeds, image_grid_thw)
    elif image_grid_thw is not None and len(image_grid_thw) > 0:
        # Models that provide grid info but aren't recognized as Qwen
        # (e.g. future Qwen variants before model_type is added to our detection)
        return _construct_qwen_image_data(image_embeds, image_grid_thw)
    else:
        # Default image handling (e.g., LLaVA, generic models)
        return {"image": image_embeds}


def _construct_qwen_image_data(
    image_embeds: torch.Tensor, image_grid_thw: Optional[List[Any]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct image data specifically for Qwen models."""
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for Qwen model.")

    grid_thw_tensor = torch.tensor(image_grid_thw)

    return {
        "image": {
            "image_embeds": image_embeds.squeeze(0),
            "image_grid_thw": grid_thw_tensor,
        }
    }


def construct_qwen_decode_mm_data(
    image_grid_thw: Optional[List[Any]],
    embeddings_shape: Optional[Any],
    request_id: str,
    *,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct schema-valid Qwen multimodal data for vLLM v1 disagg decode.

    This is a WORKAROUND (WAR) for vLLM's disaggregated multimodal decode limitations.

    Notes:
    - vLLM parses multimodal inputs and builds `mm_features` from `multi_modal_data`.
    - For Qwen VL models, the parser enforces that image data contains BOTH
      `image_embeds` and `image_grid_thw` keys.
    - In disaggregated decode, the KV cache already includes the vision context
      from prefill; decode still needs `mm_features` for mRoPE initialization.

    WAR Details:
    - We generate unique placeholder embeddings based on request_id to prevent
      incorrect prefix cache matches between different images with same dimensions.
    - Without this, zero embeddings + same image_grid_thw would create identical
      cache signatures, causing decode to incorrectly reuse cached KV from
      different images.

    Caching Caveat:
    - This WAR disables prefix cache reuse on the DECODE worker (each request
      has unique placeholder embeddings).
    - Prefix caching still works correctly on the PREFILL worker, which uses
      actual image embeddings. This is where the caching benefit matters since
      prefill does the heavy computation.
    - Decode receives KV blocks from prefill via NIXL transfer anyway, so
      decode-side prefix caching provides minimal benefit in disaggregated setup.
    """
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for Qwen model.")
    if embeddings_shape is None:
        raise ValueError("embeddings_shape is required for Qwen decode mm data.")

    # WAR: Use request_id hash as seed for unique placeholder values.
    # This prevents prefix cache from incorrectly matching different images
    # that happen to have the same dimensions (same image_grid_thw).
    # bit ops to convert request ID to somewhat unique value that fits in the dtype range
    if not hasattr(construct_qwen_decode_mm_data, "_counter"):
        construct_qwen_decode_mm_data._counter = 0
    fill_value = construct_qwen_decode_mm_data._counter
    construct_qwen_decode_mm_data._counter += 1
    max_val = (
        torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
    )
    if construct_qwen_decode_mm_data._counter > max_val:
        construct_qwen_decode_mm_data._counter = 0
    image_embeds = torch.full(
        embeddings_shape, fill_value=fill_value, dtype=dtype, device="cpu"
    )
    if image_embeds.ndim == 3:
        image_embeds = image_embeds.squeeze(0)

    return {
        "image": {
            "image_embeds": image_embeds,
            "image_grid_thw": torch.tensor(image_grid_thw),
        }
    }
