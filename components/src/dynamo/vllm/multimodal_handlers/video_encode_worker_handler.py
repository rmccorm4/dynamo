# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional

import av
import numpy as np
import torch
from vllm.engine.arg_utils import AsyncEngineArgs

import dynamo.nixl_connect as connect
from dynamo.runtime import DistributedRuntime

from ..multimodal_utils import (
    vLLMMultimodalRequest,
)
from ..multimodal_utils.video_utils import (
    calculate_frame_sampling_indices,
    get_video_metadata,
    load_video_content,
    open_video_container,
    prepare_tensor_for_rdma,
    read_video_pyav,
    resize_video_frames,
)

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8

# Default video frame parameters
DEFAULT_NUM_FRAMES_TO_SAMPLE = 8
DEFAULT_FRAME_HEIGHT = 336
DEFAULT_FRAME_WIDTH = 336


class VideoEncodeWorkerHandler:
    """Encode worker handler for video modality.

    Mirrors the structure of EncodeWorkerHandler for images but operates on
    video inputs.  On each ``generate()`` call the handler:

    1. Iterates over ``request.multimodal_inputs`` looking for ``video_url``.
    2. Downloads the video, samples frames, and resizes them.
    3. Transfers the raw uint8 pixel tensor via RDMA (NIXL) to the downstream
       PD worker using ``dynamo.nixl_connect``.
    4. Populates ``embeddings_shape`` and ``serialized_request`` on each
       ``MultiModalGroup`` entry and clears the ``video_url``.
    5. Yields the modified request as JSON so the caller can forward it.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        num_frames_to_sample: int = DEFAULT_NUM_FRAMES_TO_SAMPLE,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        frame_width: int = DEFAULT_FRAME_WIDTH,
    ) -> None:
        self.engine_args = engine_args
        self.model = self.engine_args.model
        self.min_workers = 1

        # Video processing parameters
        self.num_frames_to_sample = num_frames_to_sample
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Simple LRU-style cache for downloaded video content
        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)
        self._http_timeout = 60.0

        # NIXL connector – initialised in async_init
        self._connector: connect.Connector | None = None

    def cleanup(self) -> None:
        """No-op cleanup hook (mirrors EncodeWorkerHandler interface)."""
        pass

    async def async_init(self, runtime: DistributedRuntime) -> None:
        """Initialize the NIXL connector for RDMA transfers."""
        logger.info("Video encode worker startup started.")
        self._connector = connect.Connector()
        logger.info("Video encode worker startup completed.")

    async def _process_video(
        self, video_url: str, request_id: str
    ) -> torch.Tensor:
        """Download, decode, sample, and resize video frames.

        Returns a CPU uint8 tensor with shape ``(T, H, W, C)`` ready for RDMA
        transfer.
        """
        container: Optional[av.container.InputContainer] = None
        try:
            video_content_stream = await load_video_content(
                video_url,
                self._video_content_cache,
                self._cache_queue,
                self._http_timeout,
            )

            container = await open_video_container(video_content_stream, video_url)

            if not container or not container.streams.video:
                raise ValueError(f"No video stream found in {video_url}.")

            total_frames, duration_sec = get_video_metadata(container)

            indices = calculate_frame_sampling_indices(
                total_frames,
                self.num_frames_to_sample,
                duration_sec,
                video_url,
            )

            clip_np: np.ndarray = await read_video_pyav(container, indices)

            if clip_np.size == 0:
                raise ValueError(
                    f"Failed to extract any video frames from {video_url} "
                    f"for indices {indices.tolist()}. Clip is empty."
                )

            logger.debug(
                f"Req {request_id}: Extracted {clip_np.shape[0]} frames "
                f"from {video_url} (original shape {clip_np.shape})."
            )

            frames_tensor = torch.from_numpy(clip_np)  # (T, H, W, C)

            resized_frames = resize_video_frames(
                frames_tensor, self.frame_height, self.frame_width
            )

            return prepare_tensor_for_rdma(resized_frames, request_id)

        finally:
            if container is not None:
                await asyncio.to_thread(container.close)

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        """Process video inputs and forward the enriched request downstream.

        Registered with ``ModelInput.Tokens``.  Expects ``request`` to contain
        one or more ``MultiModalGroup`` entries with a non-None ``video_url``.
        """
        logger.debug(f"Video encode worker received raw request: {request}")

        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)

        request_id = request.request_id
        logger.debug(
            f"Video encode worker received request: {{ id: {request_id} }}."
        )

        if not request.multimodal_inputs:
            logger.warning(
                f"Req {request_id}: No multimodal_inputs; yielding request unchanged."
            )
            yield request.model_dump_json()
            return

        assert self._connector is not None, (
            "VideoEncodeWorkerHandler.async_init() must be called before generate()"
        )

        try:
            for idx, group in enumerate(request.multimodal_inputs):
                mm_input = group.multimodal_input
                if mm_input is None or mm_input.video_url is None:
                    logger.debug(
                        f"Req {request_id}: multimodal_inputs[{idx}] has no video_url, skipping."
                    )
                    continue

                video_url = mm_input.video_url
                logger.debug(
                    f"Req {request_id}: Processing video_url at index {idx}: "
                    f"{video_url[:100]}..."
                )

                tensor_for_rdma = await self._process_video(video_url, request_id)

                request.multimodal_inputs[idx].embeddings_shape = tuple(
                    tensor_for_rdma.shape
                )

                descriptor = connect.Descriptor(tensor_for_rdma)

                with await self._connector.create_readable(descriptor) as readable:
                    request.multimodal_inputs[idx].serialized_request = (
                        readable.metadata()
                    )
                    # Clear the video URL as a hint that frames are now passed as
                    # raw embeddings via RDMA.
                    request.multimodal_inputs[idx].multimodal_input.video_url = None

                    logger.debug(
                        f"Req {request_id}: RDMA transfer prepared for video at index {idx}."
                    )

                    yield request.model_dump_json()
                    await readable.wait_for_completion()

        except (FileNotFoundError, av.FFmpegError, ValueError) as e:
            logger.error(
                f"Req {request_id}: Error processing video "
                f"({type(e).__name__}): {e}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"Req {request_id}: Unexpected error processing video: {e}"
            )
            raise
