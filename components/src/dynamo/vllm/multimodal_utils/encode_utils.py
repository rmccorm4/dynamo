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

import hashlib
import logging
from typing import Any, Dict, Optional

import torch

from .model import ModelFamily, detect_model_family, is_vllm_encoder_active

logger = logging.getLogger(__name__)


def get_embedding_hash(key: str) -> str:
    """
    Generate a unique hash key for storing/retrieving image embeddings.

    Args:
        key: The base key string (e.g., image URL or identifier)
    Returns:
        A unique hash string for the given key.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def get_qwen_image_features(
    vision_encoder: torch.nn.Module, image_embeds: Dict[str, Any]
) -> torch.Tensor:
    """
    Extract image features using Qwen-style vision encoder.

    Supports both vLLM-loaded encoders (direct forward call) and
    AutoModel-loaded encoders (.get_image_features method).

    Args:
        vision_encoder: The vision encoder model
        image_embeds: Dictionary containing pixel values and grid information

    Returns:
        Processed image features tensor

    Raises:
        ValueError: If grid_thw is not provided for Qwen model
    """
    logger.debug(f"Encoding image of shape: {image_embeds['pixel_values'].shape}")

    pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)
    grid_thw = image_embeds.get("image_grid_thw")
    if grid_thw is None:
        raise ValueError("grid_thw is not provided")

    if is_vllm_encoder_active():
        # vLLM encoder path: direct forward call with list-style grid_thw
        grid_thw_list = grid_thw.tolist()
        return vision_encoder(pixel_values, grid_thw=grid_thw_list)

    # AutoModel fallback: use .get_image_features() with tensor grid_thw
    grid_thw = grid_thw.to(vision_encoder.device)
    logger.debug(f"Qwen grid_thw shape: {grid_thw.shape}")
    return vision_encoder.get_image_features(pixel_values, grid_thw)


def encode_image_embeddings(
    model_name: str,
    image_embeds: Dict[str, Any],
    vision_encoder: torch.nn.Module,
    projector: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Encode image embeddings using the appropriate model-specific encoder.

    Routing is determined by the model family (detected from HuggingFace config)
    and the presence of a projector module:
    - Qwen VL: uses grid_thw-aware encoding via get_qwen_image_features
    - LLaVA-style (projector present): vision_encoder + projector pipeline
    - Generic (no projector): direct forward call on vision_encoder

    Args:
        model_name: The model identifier
        image_embeds: Dictionary containing processed image data
        vision_encoder: The vision encoder module
        projector: The multimodal projector (required for LLaVA-style models)

    Returns:
        Encoded embeddings tensor with normalized shape

    Raises:
        ValueError: If projector is missing for LLaVA models
    """
    family = detect_model_family(model_name)

    with torch.no_grad():
        if family == ModelFamily.QWEN_VL:
            embeddings = get_qwen_image_features(vision_encoder, image_embeds)

        elif projector is not None:
            # LLaVA-style: vision encoder + projector pipeline
            pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)
            vision_outputs = vision_encoder(pixel_values)

            if hasattr(vision_outputs, "last_hidden_state"):
                embeddings = projector(vision_outputs.last_hidden_state)
            else:
                # Some vision encoders return the hidden states directly
                embeddings = projector(vision_outputs)

        else:
            # Generic: direct forward call
            pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)
            vision_outputs = vision_encoder(pixel_values)

            if hasattr(vision_outputs, "last_hidden_state"):
                embeddings = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, (tuple, list)):
                embeddings = vision_outputs[0]
            else:
                embeddings = vision_outputs

        # Normalize output shape
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

    return embeddings


def get_encoder_components(
    model_name: str, vision_model: torch.nn.Module
) -> tuple[Any, Optional[Any]]:
    """
    Get the appropriate vision encoder and projector components for a given model.

    Detection is based on model family (from HuggingFace config) and model
    attributes, rather than a hardcoded model list.

    Args:
        model_name: The model identifier
        vision_model: The loaded vision model (full model from vLLM or AutoModel)

    Returns:
        Tuple of (vision_encoder, projector) where types depend on the model

    Raises:
        NotImplementedError: If vision encoder components cannot be determined
    """
    family = detect_model_family(model_name)

    if family == ModelFamily.QWEN_VL:
        if is_vllm_encoder_active():
            # vLLM loads the full model; extract .visual for direct forward calls
            visual = getattr(vision_model, "visual", None)
            if visual is None:
                logger.warning(
                    "Expected 'visual' attribute on vLLM-loaded Qwen VL model '%s', "
                    "using model directly.",
                    model_name,
                )
                return vision_model, None
            return visual, None
        else:
            # AutoModel: use full model for .get_image_features() calls
            return vision_model, None

    # LLaVA-style: look for vision_tower + projector
    vision_tower = getattr(vision_model, "vision_tower", None)
    projector = getattr(vision_model, "multi_modal_projector", None)
    if vision_tower is not None:
        return vision_tower, projector

    # Generic: probe for common vision component attribute names
    for attr_name in ("visual", "vision_model", "vision_encoder", "image_encoder"):
        component = getattr(vision_model, attr_name, None)
        if component is not None and isinstance(component, torch.nn.Module):
            logger.info(
                "Using '%s' as vision encoder for model '%s'", attr_name, model_name
            )
            return component, None

    raise NotImplementedError(
        f"Could not extract vision encoder from model '{model_name}' "
        f"(type: {type(vision_model).__name__}, family: {family.value}). "
        f"If this model is supported by vLLM, please file an issue."
    )
