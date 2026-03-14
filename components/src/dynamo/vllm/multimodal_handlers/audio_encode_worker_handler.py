# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from typing import AsyncIterator

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.common.multimodal import AudioLoader
from dynamo.runtime import DistributedRuntime

from ..handlers import build_sampling_params
from ..multimodal_utils import (
    MyRequestOutput,
    MultiModalInput,
    MultiModalGroup,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8

AUDIO_URL_KEY = "audio_url"


class AudioEncodeWorkerHandler:
    """Standalone audio encode worker for Qwen2-Audio models.

    Mirrors the structure of EncodeWorkerHandler (image encode worker) but
    operates on audio inputs.  At init time it loads AutoProcessor and
    Qwen2AudioForConditionalGeneration then, on every generate() call, it:

    1. Parses ``audio_url`` from the incoming Rust SDK request
       (``ModelInput.Tokens`` format:
       ``{"token_ids": [...], "multi_modal_data": {"audio_url": [...]}}``)
    2. Downloads and encodes the audio using the model's audio tower +
       ``multi_modal_projector`` via :meth:`get_audio_embeddings`.
    3. Transfers the resulting float16 tensor to the downstream PD worker via
       RDMA (``dynamo.nixl_connect`` ``Connector`` / ``Descriptor`` /
       ``create_readable`` pattern).
    4. Populates ``multimodal_inputs[0].embeddings_shape`` and
       ``multimodal_inputs[0].serialized_request`` on a ``vLLMMultimodalRequest``
       (audio_url cleared), forwards it to the downstream PD worker via
       ``pd_worker_client``, and streams responses back to the caller.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        pd_worker_client,
    ) -> None:
        self.engine_args = engine_args
        self.model = self.engine_args.model
        self.pd_worker_client = pd_worker_client
        self.default_sampling_params = (
            self.engine_args.create_model_config().get_diff_sampling_param()
        )

        self.audio_loader = AudioLoader(cache_size=CACHE_SIZE_MAXIMUM)

        logger.info("Loading audio processor from %s ...", self.model)
        self.audio_processor = AutoProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )

        logger.info("Loading Qwen2-Audio model from %s ...", self.model)
        self.audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model, device_map="auto", dtype=torch.float16
        ).eval()

        self._connector: connect.Connector | None = None

    # ── Audio encoding ───────────────────────────────────────────────

    def get_audio_embeddings(self, audio_features) -> torch.Tensor:
        """Run audio through the model's audio tower + multi_modal_projector.

        Returns the masked audio feature tensor (shape: ``[num_tokens, embed_dim]``).
        """
        input_features, feature_attention_mask = (
            audio_features.input_features,
            audio_features.feature_attention_mask,
        )
        with torch.no_grad():
            (
                audio_feat_lengths,
                audio_output_lengths,
            ) = self.audio_model.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1

            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feat_lengths.dtype,
                    device=audio_feat_lengths.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len
            )
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len
            ).expand(batch_size, 1, max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_model.audio_tower.conv1.weight.dtype,
                device=self.audio_model.audio_tower.conv1.weight.device,
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.audio_model.audio_tower(
                input_features, attention_mask=audio_attention_mask
            )
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features_out = self.audio_model.multi_modal_projector(
                selected_audio_feature
            )

            num_audios, max_audio_tokens, embed_dim = audio_features_out.shape
            audio_features_mask = torch.arange(
                max_audio_tokens, device=audio_output_lengths.device
            )[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
            audio_features_out = audio_features_out[audio_features_mask]

            return audio_features_out

    # ── Request parsing ──────────────────────────────────────────────

    def _parse_frontend_request(
        self, raw_request: dict
    ) -> tuple[vLLMMultimodalRequest, str | None]:
        """Parse a raw Rust SDK request dict into a vLLMMultimodalRequest and audio URL.

        The Rust SDK sends
        ``{"token_ids": [...], "multi_modal_data": {"audio_url": [{"Url": "https://..."}]}}``
        when registered as ``ModelInput.Tokens``.
        """
        request_id = str(uuid.uuid4().hex)

        audio_url: str | None = None
        mm_data = raw_request.get("multi_modal_data") or {}
        for item in mm_data.get(AUDIO_URL_KEY, []):
            if isinstance(item, dict) and "Url" in item:
                audio_url = item["Url"]
                break

        sampling_params = build_sampling_params(
            raw_request, self.default_sampling_params
        )

        request = vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(
                prompt_token_ids=raw_request["token_ids"]
            ),
            sampling_params=sampling_params,
            request_id=request_id,
            model=raw_request.get("model"),
        )

        return request, audio_url

    # ── Lifecycle ────────────────────────────────────────────────────

    def cleanup(self) -> None:
        pass

    async def async_init(self, runtime: DistributedRuntime) -> None:
        """Initialize the NIXL connector for RDMA transfers."""
        logger.info("Audio encode worker startup started.")
        self._connector = connect.Connector()
        logger.info("Audio encode worker startup completed.")

    # ── Main entry point ─────────────────────────────────────────────

    async def generate(self, raw_request: dict, context) -> AsyncIterator[str]:
        """Encode audio and forward the resulting embeddings to the PD worker.

        Steps:

        1. Parse the incoming request and extract the audio URL.
        2. Download audio via :class:`~dynamo.common.multimodal.AudioLoader`.
        3. Pre-process with ``AutoProcessor``.
        4. Run through audio tower + projector via :meth:`get_audio_embeddings`.
        5. Transfer embeddings via RDMA (``create_readable`` pattern).
        6. Forward ``vLLMMultimodalRequest`` with ``multimodal_inputs[0]``
           populated (``embeddings_shape``, ``serialized_request``,
           ``audio_url`` cleared) to the downstream PD worker.
        7. Stream PD worker responses back to the caller.
        """
        if not isinstance(raw_request, dict):
            if isinstance(raw_request, str):
                import json as _json

                raw_request = _json.loads(raw_request)
            else:
                raise ValueError(
                    f"AudioEncodeWorkerHandler.generate() expected a dict, "
                    f"got {type(raw_request)}"
                )

        request, audio_url = self._parse_frontend_request(raw_request)
        request_id = request.request_id
        logger.debug("Received audio encode request: { id: %s }.", request_id)

        if not audio_url:
            raise ValueError(
                f"No audio_url found in request {request_id}. "
                "The request must contain multi_modal_data.audio_url."
            )

        assert self._connector is not None, (
            "AudioEncodeWorkerHandler.async_init() must be called before generate()"
        )

        try:
            # 1. Download audio
            audio, sr = await self.audio_loader.load_audio(audio_url)

            # 2. Pre-process
            audio_features = self.audio_processor(
                text="test<|AUDIO|>",
                audio=audio,
                return_tensors="pt",
                padding=False,
            )

            # 3. Encode (blocking – runs on GPU via audio_model)
            with torch.no_grad():
                audio_embeddings = self.get_audio_embeddings(audio_features)

            # 4. Transfer via RDMA and forward to PD worker
            descriptor = connect.Descriptor(audio_embeddings)
            with await self._connector.create_readable(descriptor) as readable:
                # Store RDMA transfer metadata and embedding shape in a
                # MultiModalGroup entry (audio_url cleared to signal the audio
                # is now carried as pre-computed embeddings, not a URL).
                audio_group = MultiModalGroup(
                    multimodal_input=MultiModalInput(audio_url=None),
                    embeddings_shape=tuple(audio_embeddings.shape),
                    serialized_request=readable.metadata(),
                )
                request.multimodal_inputs = [audio_group]

                logger.debug("Forwarding request %s to PD worker.", request_id)
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )

                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()

        except Exception as e:
            logger.error(
                "Error processing audio encode request %s: %s", request_id, e
            )
            raise
