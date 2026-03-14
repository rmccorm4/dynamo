#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check and install required dependencies for audio multimodal models
echo "Checking audio multimodal dependencies..."
DEPS_MISSING=false

# Check for accelerate
if ! python -c "import accelerate" &> /dev/null; then
    echo "  accelerate not found"
    DEPS_MISSING=true
else
    echo "  ✓ accelerate is installed"
fi

# Check for vllm with audio support
if ! python -c "import vllm" &> /dev/null; then
    echo "  vllm not found"
    DEPS_MISSING=true
else
    # Check if audio dependencies are available (librosa is a key audio dependency)
    if ! python -c "import librosa" &> /dev/null; then
        echo "  vllm audio dependencies not found"
        DEPS_MISSING=true
    else
        echo "  ✓ vllm with audio support is installed"
    fi
fi

# Install missing dependencies
if [ "$DEPS_MISSING" = true ]; then
    echo "Installing missing dependencies..."
    pip install 'vllm[audio]' accelerate
    echo "Dependencies installed successfully"
else
    echo "All required dependencies are already installed"
fi

# run ingress
python -m dynamo.frontend --http-port 8000 &

# run E/P/D workers
CUDA_VISIBLE_DEVICES=0 python3 components/audio_encode_worker.py --model $MODEL_NAME &
DYN_VLLM_KV_EVENT_PORT=20081 VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=1 python3 components/worker.py --model $MODEL_NAME --worker-type prefill --enable-disagg &
DYN_VLLM_KV_EVENT_PORT=20082 VLLM_NIXL_SIDE_CHANNEL_PORT=20099 CUDA_VISIBLE_DEVICES=2 python3 components/worker.py --model $MODEL_NAME --worker-type decode --enable-disagg &

# Wait for all background processes to complete
wait
