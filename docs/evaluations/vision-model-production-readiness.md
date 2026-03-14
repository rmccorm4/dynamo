# Vision Language Model (VLM) Production Readiness Evaluation

**Date**: 2026-03-14
**Scope**: Evaluate the production readiness of serving multimodal vision language models in Dynamo

## Executive Summary

Dynamo provides **substantial VLM support** with a well-architected multimodal pipeline spanning three backends (vLLM, TRT-LLM, SGLang). **Image-based VLM serving is production-ready** for aggregated deployments and approaching production-ready for fully disaggregated (E/P/D) serving. However, several gaps remain in observability, model coverage, and performance at scale that should be addressed before broad production rollout.

**Overall Readiness: 7/10** — Production-viable for image VLMs in aggregated mode; disaggregated mode ready for confirmed models (LLaVA); gaps remain in monitoring, multi-model coverage, and advanced modalities.

---

## 1. Architecture Overview

Dynamo supports three deployment patterns for VLMs:

| Pattern | Description | Maturity |
|---------|-------------|----------|
| **EPD (Aggregated)** | Encode + Prefill + Decode in one worker | Production-ready |
| **E/PD** | Separate encoder; Prefill+Decode combined | Production-ready |
| **E/P/D** | Fully disaggregated: Encode → Prefill → Decode | Production-ready (LLaVA); other models unconfirmed |

Inter-component communication uses NIXL for zero-copy RDMA embedding and KV cache transfer. The Rust frontend handles OpenAI-compatible API parsing, media downloading/decoding, and tokenization before dispatching to Python backend workers.

---

## 2. What Works Well

### 2.1 OpenAI-Compatible API (Strong)
- Full support for the OpenAI vision API format (content arrays with `image_url`, `video_url`, `audio_url`)
- Supports HTTP/HTTPS URLs, base64 data URLs, and pre-computed embedding files
- Response types also support multimodal content parts
- **Files**: `lib/async-openai/src/types/chat.rs`, `lib/llm/src/protocols/openai/chat_completions.rs`

### 2.2 Media Processing Pipeline (Strong)
- Rust-native image decoding (PNG, JPEG, BMP, WebP, GIF, TIFF) with configurable limits
- Secure media fetcher with domain whitelisting, IP access control, and URL scheme validation
- NIXL registration for GPU-direct memory access of decoded media
- **Files**: `lib/llm/src/preprocessor/media/`

### 2.3 Encoder Disaggregation (Good)
- Dedicated encode worker architecture with vision model extraction (ViT + projector)
- Three embedding transfer modes: LOCAL, NIXL_WRITE, NIXL_READ
- Embedding cache (LRU) to skip re-encoding repeated images
- **Files**: `components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py`

### 2.4 Multimodal KV Routing (Good)
- Block-level multimodal metadata with content hashing (XXH3)
- KV router accounts for image-bearing blocks when computing cache overlap
- Repeated images route to workers with highest KV cache hit potential
- **Files**: `lib/kv-router/src/protocols.rs`, `docs/features/multimodal/multimodal-kv-routing.md`

### 2.5 Security Model (Strong)
- Explicit `--enable-multimodal` flag required on all multimodal workers
- Media fetcher has domain whitelisting, IP blocking, port control
- File size limits enforced for embeddings (50 MB default, configurable)
- Local path access restricted to configured directories

---

## 3. Gaps and Risks

### 3.1 Critical Gaps

#### 3.1.1 No Vision-Specific Observability
**Severity: High** — No metrics exist for:
- Vision encoding time per request
- Embedding cache hit/miss rates
- NIXL embedding transfer latency
- Concurrent request latency scaling (known to degrade — see WIP comment in `encode_worker_handler.py:40-44`)
- Memory usage tracking for media decoding

**Impact**: Operators cannot monitor, alert on, or optimize VLM-specific bottlenecks in production.

#### 3.1.2 Embedding Transfer Latency Scales Poorly
**Severity: High** — Code comment in `encode_worker_handler.py:40-44` explicitly notes: embedding transfer latency increases with concurrent requests, and NIXL transfer scales worse than local transfer. This is marked as WIP with no resolution.

**Impact**: Under production load, VLM serving throughput may degrade non-linearly as concurrency increases.

#### 3.1.3 Hardcoded Model Support List
**Severity: Medium** — Supported VLM architectures are maintained as an explicit list in `multimodal_utils/model.py:36` with a TODO to replace with dynamic detection via HuggingFace config.

**Impact**: Adding new VLM architectures requires code changes rather than configuration.

### 3.2 Model Coverage Gaps

| Model Family | Aggregated | E/PD | E/P/D | Notes |
|-------------|-----------|------|-------|-------|
| LLaVA 1.5 | vLLM ✅ TRT-LLM ✅ | ✅ | ✅ | Best-tested path |
| Qwen2.5-VL | vLLM ✅ TRT-LLM ✅ | ✅ | ❓ Unconfirmed | Multi-image handling unclear (FIXME in code) |
| Qwen3-VL | vLLM ✅ | ✅ | ❓ Unconfirmed | Newer model family |
| Llama 4 Maverick/Scout | vLLM ✅ TRT-LLM ✅ | ✅ (native encoding) | ❓ | Uses native encoding, not separate encoder |
| LLaVA 1.6 | TRT-LLM ✅ | ✅ | ❓ | Known TRT-LLM version incompatibility |
| SGLang (any VLM) | ✅ | ✅ | ✅ | No data URL support; no pre-computed embeddings |

### 3.3 Backend-Specific Limitations

**vLLM**:
- Aggregated embedding cache requires unreleased vLLM patches (PRs #34182, #34783)
- MM KV routing awaiting upstream vLLM PR #33304
- Qwen-specific embedding splitting logic needs abstraction (`encode_worker_handler.py:241`)

**TRT-LLM**:
- NIXL KV cache transfer is beta and x86_64-only (no ARM64)
- Known model crash with `llava-v1.6-mistral-7b-hf` on TRT-LLM 1.2.0rc6.post1
- CUDA IPC handle extraction performance needs optimization (tracked as DIS-1398)

**SGLang**:
- No data URL (base64) support — only HTTP/HTTPS URLs
- No pre-computed embedding file support
- No MM KV routing support
- No embedding cache support

### 3.4 Operational Gaps

| Gap | Severity | Details |
|-----|----------|---------|
| Frontend media decoding incompatible with `Dockerfile.frontend` | Medium | Lightweight frontend image lacks NIXL/UCX dependencies |
| Frontend requires GPU node | Medium | CPU-only frontend nodes fail with UCX initialization error |
| No multi-turn multimodal conversation support | Medium | TODO in `chat_message_utils.py:20` |
| Hardcoded embedding dtype | Low | `torch.uint8` for video, `torch.float16` for images — not configurable |
| Encoder cache size hardcoded to 8 | Low | `CACHE_SIZE_MAXIMUM = 8` in `encode_worker_handler.py:38` |

---

## 4. Modality Support Matrix

| Modality | vLLM | TRT-LLM | SGLang | Frontend (Rust) |
|----------|------|---------|--------|-----------------|
| **Image** | ✅ Production | ✅ Production | ✅ Production | ✅ Production |
| **Video** | 🧪 Experimental | ❌ | ❌ | 🧪 Requires ffmpeg feature |
| **Audio** | 🧪 Experimental | ❌ | ❌ | ❌ Not implemented |
| **Pre-computed Embeddings** | ✅ | ✅ | ❌ | N/A |
| **Data URLs (base64)** | ✅ | ✅ | ❌ | ✅ |

---

## 5. Test Coverage Assessment

### What's Tested
- E2E multimodal KV routing with TRT-LLM and vLLM (`tests/mm_router/`)
- vLLM multimodal PD worker handler (`tests/` in vLLM component)
- Embedding cache integration
- Disaggregated determinism validation (`tests/kvbm_integration/`)
- Router correctness with mock and real backends

### What's Missing
- Multi-image request handling across all model families
- Multi-turn multimodal conversations
- SGLang multimodal disaggregated (E/P/D)
- NIXL failure recovery and retry behavior
- Concurrent request scaling under load (the known regression)
- ARM64 architecture testing
- Frontend media decoding across container configurations
- Video and audio processing (experimental features)

---

## 6. Performance Characteristics

From documented benchmarks:
- **RDMA is critical**: Without RDMA, KV transfer causes ~40x TTFT degradation (355ms → 10+ seconds)
- **Disaggregated throughput**: ~1.38x improvement over aggregated (446.85 vs 322.69 tokens/s/gpu on H200)
- **Known regression**: Embedding transfer latency scales poorly with concurrent requests (unresolved WIP)

---

## 7. Recommendations

### P0 — Required Before Production VLM Rollout
1. **Add vision-specific metrics**: Encoding time, embedding cache hit rate, NIXL transfer latency, per-modality request counts
2. **Investigate and fix embedding transfer scaling**: The documented WIP regression in `encode_worker_handler.py` is a production blocker under load
3. **Validate Qwen2.5-VL and Qwen3-VL in disaggregated mode**: These are popular production models with unconfirmed disaggregated support

### P1 — Important for Production Quality
4. **Replace hardcoded model list with dynamic detection**: Use HuggingFace model config to auto-detect VLM capabilities
5. **Add SGLang data URL support**: Base64 image input is a common production pattern
6. **Document container requirements**: Clarify GPU/NIXL/UCX dependencies for frontend media decoding
7. **Add multi-turn multimodal conversation support**: Required for chat-style VLM applications
8. **Make embedding cache size configurable**: Current hardcoded limit of 8 is too small for production

### P2 — Nice to Have
9. **Abstract Qwen-specific embedding logic**: Remove model-specific FIXMEs in encode worker
10. **Add ARM64 support for NIXL KV transfer**: Currently beta and x86_64-only
11. **Stabilize video support**: Move from experimental to production-ready on at least one backend

---

## 8. Conclusion

Dynamo's VLM serving infrastructure is architecturally sound with a clean separation between Rust frontend preprocessing and Python backend integration. The three-tier disaggregated architecture (E/P/D) is a differentiating capability that enables independent scaling of compute-bound (encoding, prefill) and memory-bound (decode) phases.

**For image-only VLMs in aggregated mode**, the system is production-ready today with well-tested models (LLaVA, Qwen2.5-VL). **For fully disaggregated VLM serving**, the system is production-ready for LLaVA and likely ready for other models pending validation. The primary risks are the lack of vision-specific observability and the documented embedding transfer scaling regression under concurrent load.

The recommendation is to proceed with production deployment for aggregated image VLM workloads while addressing P0 items before scaling disaggregated VLM serving.
