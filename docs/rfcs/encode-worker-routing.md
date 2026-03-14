# RFC: Encode Worker Routing via the Dynamo Router

## Status: Draft

## Problem

Today, the Encode worker is invisible to the Dynamo router. The Prefill (or PD)
worker discovers Encode workers via `runtime.endpoint("encoder.generate").client()`
and dispatches to them directly with `client.round_robin()`. This creates several
issues:

1. **Extra data hops**: Image URLs or base64 data travel
   Frontend -> Prefill -> Encode, then embeddings travel Encode -> Prefill via
   NIXL. In E/P/D, the Prefill worker is a middleman that doesn't need to see
   raw image data at all.

2. **No router-aware load balancing**: Encode worker selection is blind
   round-robin from the Prefill worker. The router's KV-aware scheduling,
   fault detection, and metrics integration don't apply.

3. **Tight coupling**: The Prefill worker must know about Encode workers,
   manage batching across them, and handle NIXL transfer lifecycle.

## Current Data Flow

```
                          E/PD topology
  ┌──────────┐  tokens+URLs  ┌─────────┐  URLs     ┌─────────┐
  │ Frontend │──────────────>│   PD    │─────────->│ Encode  │
  │  (Rust)  │               │ Worker  │<──────────│ Worker  │
  └──────────┘               │         │ NIXL      └─────────┘
                             │         │ embeddings
                             └─────────┘

                          E/P/D topology
  ┌──────────┐  tokens+URLs  ┌─────────┐  URLs     ┌─────────┐
  │ Frontend │──────────────>│ Prefill │─────────->│ Encode  │
  │  (Rust)  │               │ Worker  │<──────────│ Worker  │
  └──────────┘               │         │ NIXL      └─────────┘
                             │         │ embeddings
                             │         │──KV──────>┌─────────┐
                             └─────────┘           │ Decode  │
                                                   │ Worker  │
                                                   └─────────┘
```

**Problems with current flow:**
- Frontend downloads/decodes images (Rust `media::loader`), then serializes
  them as RDMA descriptors or URLs in `multi_modal_data`. The PD worker
  receives the full token+multimodal payload, extracts URLs, and re-sends
  them to the Encode worker. The image data crosses 3 process boundaries.
- NIXL transfer lifecycle (staging, completion, release) is managed inside
  `prefill_worker_utils.py` which increases Prefill worker complexity.

## Proposed Design: NIXL-Handle-First Routing

### Core Idea

The Encode worker becomes a first-class routable component. The Frontend (or a
thin routing layer) sends image data directly to the Encode worker. The Encode
worker returns a lightweight **NIXL handle** (not the embeddings themselves).
The handle is attached to the request that flows to Prefill. Prefill reads
embeddings from the handle via zero-copy RDMA.

### Data Flow

```
                     E/PD topology (proposed)
  ┌──────────┐  URLs   ┌─────────┐
  │ Frontend │────────>│ Encode  │ (1) encode, stage NIXL readable
  │  (Rust)  │<────────│ Worker  │     return NIXL handle + metadata
  └──────────┘ handle  └─────────┘
       │
       │ tokens + NIXL handle (small JSON)
       v
  ┌──────────┐                         ┌─────────┐
  │    PD    │──NIXL READ (zero-copy)─>│ Encode  │ (2) RDMA pull
  │  Worker  │                         │ Worker  │     (GPU memory)
  └──────────┘                         └─────────┘

                     E/P/D topology (proposed)
  ┌──────────┐  URLs   ┌─────────┐
  │ Frontend │────────>│ Encode  │ (1) encode, stage NIXL readable
  │  (Rust)  │<────────│ Worker  │     return NIXL handle
  └──────────┘ handle  └─────────┘
       │
       │ tokens + NIXL handle
       v
  ┌──────────┐                         ┌─────────┐
  │ Prefill  │──NIXL READ (zero-copy)─>│ Encode  │ (2) RDMA pull
  │ Worker   │                         │ Worker  │
  └──────────┘                         └─────────┘
       │
       │ KV cache via NIXL
       v
  ┌──────────┐
  │  Decode  │ (3) No multimodal data needed
  │  Worker  │
  └──────────┘
```

### Why This Is Optimal

| Metric | Current | Proposed |
|--------|---------|----------|
| Image data hops | Frontend -> PD -> Encode | Frontend -> Encode (1 hop) |
| Embedding data hops | Encode -> PD (NIXL push) | Encode <- PD (NIXL pull, same cost) |
| Serialized payload size to PD | tokens + image URLs | tokens + NIXL handle (~500B) |
| Prefill worker complexity | Manages encode dispatch + NIXL lifecycle | Just reads handle |
| Router visibility of Encode | None | Full (discovery, metrics, fault detection) |

The key optimization: **NIXL READ mode** is already implemented
(`NixlReadEmbeddingSender` / `NixlReadEmbeddingReceiver`). In this mode, the
Encode worker stages a `ReadableOperation` and returns an `RdmaMetadata` handle
(~500 bytes: NIXL agent metadata + memory descriptor + notification UUID). The
Prefill worker uses `connector.begin_read(handle, local_descriptor)` to pull
embeddings via zero-copy RDMA. The image data never flows through the Prefill
worker at all.

## Implementation

### Phase 1: Register Encode Worker as a Routable Endpoint

The Encode worker already registers at `{namespace}.encoder.generate`. We make
the Frontend aware of it as a preprocessing step.

#### 1a. Encode Worker: Return Handle Instead of Full Embedding

The Encode worker's `generate()` method currently yields a serialized
`vLLMMultimodalRequest` with embedded transfer metadata. Instead, it should
yield a compact response containing just the NIXL handle and embedding metadata.

```python
# components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py

@dataclass
class EncodeResult:
    """Compact result from the Encode worker - just metadata, no embeddings."""
    nixl_handles: list[dict]        # RdmaMetadata per image (NIXL READ handle)
    image_grid_thws: list[list]     # Grid dimensions per image (Qwen)
    embeddings_shapes: list[tuple]  # Shape per image embedding


class EncodeWorkerHandler:

    async def generate(
        self, request: EncodeWorkerRequest, context
    ) -> AsyncIterator[str]:
        """Encode images and return NIXL handles (not embeddings)."""
        # ... (existing image load + preprocess + vision encode logic) ...

        # Stage each embedding as a NIXL readable (existing NixlReadEmbeddingSender)
        nixl_handles = []
        for embedding in splitted_embeddings:
            transfer_req, completion_future = await self.embedding_sender.send_embeddings(
                embedding.unsqueeze(0), stage_embeddings=True
            )
            nixl_handles.append(transfer_req.serialized_request)
            self.send_complete_queue.put_nowait((completion_future, embedding))

        result = EncodeResult(
            nixl_handles=nixl_handles,
            image_grid_thws=[
                [image_grid_thw[i]] if image_grid_thw else []
                for i in range(len(nixl_handles))
            ],
            embeddings_shapes=[
                tuple(splitted_embeddings[i].unsqueeze(0).shape)
                for i in range(len(nixl_handles))
            ],
        )
        yield json.dumps(asdict(result))
```

#### 1b. Frontend Routes to Encode Worker Before Prefill

The Frontend (or a new routing middleware) intercepts multimodal requests:

```python
# Conceptual middleware (could be Rust or Python)
# This runs in the Frontend or a thin "multimodal router" component

async def route_multimodal_request(request, context):
    """Route multimodal request through Encode worker, then to Prefill."""

    image_urls = extract_image_urls(request)
    if not image_urls:
        # Text-only: route directly to Prefill/PD
        return await prefill_client.round_robin(request, context=context)

    # Step 1: Send images to Encode worker (round-robin across Encode pool)
    encode_request = EncodeWorkerRequest(
        image_urls=image_urls,
        request_id=request["request_id"],
    )
    encode_stream = await encode_client.round_robin(
        encode_request.model_dump_json(), context=context
    )
    async for response in encode_stream:
        encode_result = EncodeResult(**json.loads(response.data()))

    # Step 2: Replace image URLs with NIXL handles in the original request
    request["multi_modal_data"] = {
        "nixl_handles": encode_result.nixl_handles,
        "image_grid_thws": encode_result.image_grid_thws,
        "embeddings_shapes": encode_result.embeddings_shapes,
    }
    # Remove raw image data - Prefill doesn't need it
    request["multi_modal_data"].pop("image_url", None)

    # Step 3: Route to Prefill/PD worker (can use KV-aware routing)
    return await prefill_client.route(request, context=context)
```

#### 1c. Prefill Worker Reads Embeddings from NIXL Handle

The Prefill worker receives the request with NIXL handles instead of image
URLs. It reads embeddings directly via RDMA:

```python
# components/src/dynamo/vllm/multimodal_handlers/multimodal_pd_worker_handler.py

async def _load_multimodal_data(
    self, request: dict, request_id: str, context=None
) -> dict[str, Any]:
    """Read embeddings from NIXL handles attached to the request."""

    mm_data = request.get("multi_modal_data", {})
    nixl_handles = mm_data.get("nixl_handles")
    if not nixl_handles:
        return defaultdict(list)

    embeddings_shapes = mm_data["embeddings_shapes"]
    image_grid_thws = mm_data.get("image_grid_thws", [])

    # Read each embedding via NIXL (zero-copy RDMA pull)
    multi_modal_data = defaultdict(list)
    for i, handle in enumerate(nixl_handles):
        embedding = await self._read_embedding_from_handle(
            handle, embeddings_shapes[i]
        )
        _accumulate_embeddings(
            multi_modal_data,
            self.config.model,
            self.EMBEDDINGS_DTYPE,
            embedding,
            image_grid_thws[i] if image_grid_thws else None,
        )

    return multi_modal_data

async def _read_embedding_from_handle(
    self, nixl_handle: dict, shape: tuple
) -> torch.Tensor:
    """Read an embedding tensor from a NIXL RDMA handle."""
    embedding = torch.empty(shape, dtype=self.EMBEDDINGS_DTYPE, device="cpu")
    descriptor = connect.Descriptor(embedding)
    read_op = await self._connector.begin_read(nixl_handle, descriptor)
    await read_op.wait_for_completion()
    return embedding
```

### Phase 2: Router-Integrated Encode Scheduling

Once the Encode worker is a first-class endpoint, the router can make
intelligent scheduling decisions:

#### Encode-Aware Routing Strategies

```
Strategy 1: Round-Robin (default, simplest)
  Router round-robins across Encode workers, then round-robins across
  Prefill workers. Good for homogeneous GPU pools.

Strategy 2: Locality-Aware
  Route to an Encode worker on the same node as the target Prefill worker.
  Reduces NIXL transfer to intra-node (PCIe/NVLink) instead of inter-node
  (InfiniBand). Requires node topology metadata in worker registration.

Strategy 3: Sticky Image Hashing
  Hash the image URL to select the Encode worker. This maximizes the
  Encode worker's local embedding cache hit rate. Combined with returning
  a cached NIXL handle, repeat images skip encoding entirely.
```

#### Encode Worker Discovery and Health

```yaml
# The Encode worker registers like any other Dynamo endpoint:
# namespace: dynamo
# component: encoder
# endpoint: generate
# worker_type: encoder  (new label for routing metadata)

# Deployment config (dynamo.yaml)
services:
  Frontend:
    type: http
    routes_to: [Encoder, PrefillWorker]  # NEW: Frontend knows about Encoder

  Encoder:
    type: worker
    endpoint: "encoder.generate"
    gpu: true
    replicas: 2

  PrefillWorker:
    type: worker
    endpoint: "llm.generate"
    gpu: true
    replicas: 4
    reads_from: [Encoder]  # Gets NIXL handles, not image data

  DecodeWorker:
    type: worker
    endpoint: "decoder.generate"
    gpu: true
    replicas: 4
```

### Phase 3 (Future): Rust-Native Encode Routing

The Rust frontend already has `media::loader` which downloads and decodes
images, and `media::rdma` which stages decoded pixels as NIXL descriptors.
The natural evolution:

1. Frontend's Rust preprocessor extracts image URLs from the request
2. Frontend dispatches to Encode worker via Rust `PushRouter::round_robin()`
3. Frontend receives NIXL handle back
4. Frontend attaches handle to the tokenized request
5. Frontend routes to Prefill via KV-aware router

This keeps the entire image data path in Rust (no Python serialization overhead)
and the NIXL handle is just a ~500B JSON blob.

## Performance Analysis

### Data Movement Comparison

For a request with 1 image (1024x1024 JPEG, ~200KB compressed, ~3MB decoded):

| Stage | Current | Proposed |
|-------|---------|----------|
| Frontend receives image URL | URL string (~100B) | URL string (~100B) |
| Frontend -> worker | URL in JSON (~100B) | URL to Encode (~100B) |
| Image download | In Encode worker | In Encode worker |
| Image preprocess | In Encode worker | In Encode worker |
| Vision encode | In Encode worker | In Encode worker |
| Embedding result | ~4MB via NIXL push to PD | ~500B NIXL handle to Frontend |
| Handle to PD/Prefill | N/A (PD already has embedding) | ~500B in JSON request |
| Embedding to PD/Prefill | Already there | ~4MB via NIXL pull (same cost) |

**Net improvement**: The Prefill worker never sees image URLs, never manages
Encode worker dispatch, never manages NIXL sender lifecycle. The embedding
transfer cost is identical (NIXL push vs pull is symmetric). The payload
to Prefill is ~500B instead of containing image URLs.

### Latency Impact

- **Additional hop**: Frontend -> Encode -> Frontend adds one round-trip
  (~0.5-1ms network + serialization). But this replaces the current
  PD -> Encode -> PD round-trip, so it's roughly the same.
- **Parallelism**: The Frontend can dispatch to Encode while the Prefill
  worker is busy with other requests. Currently the Prefill worker blocks
  on Encode dispatch.
- **Cache locality**: With sticky URL hashing, the Encode worker's embedding
  cache hit rate improves significantly for repeat images.

### When NOT to Use This Pattern

- **Co-located Encode+Prefill on same GPU**: If running both on the same
  device (e.g., vLLM `mm_encoder_only` shares the GPU), the NIXL overhead
  is unnecessary. Use the current direct-call path.
- **Very small images**: If embedding transfer cost dominates over encoding
  cost, the handle indirection adds complexity without benefit.

## Migration Path

1. **Phase 1A**: Add `EncodeResult` response type and NIXL-handle-returning
   mode to `EncodeWorkerHandler`. Keep existing mode as default.
2. **Phase 1B**: Add routing middleware (Python) that dispatches to Encode
   then Prefill. Gate behind `--route-encode-via-frontend` flag.
3. **Phase 1C**: Update `MultimodalPDWorkerHandler` to accept NIXL handles
   in `multi_modal_data` alongside the existing URL-based path.
4. **Phase 2**: Integrate Encode worker into the Rust router's service
   discovery. Add encode-aware scheduling strategies.
5. **Phase 3**: Move Encode dispatch into the Rust frontend's preprocessor
   pipeline.

## Open Questions

1. **Handle lifetime management**: The Encode worker must keep embeddings in
   GPU memory until the Prefill worker reads them. With NIXL READ, the
   `ReadableOperation` holds the memory. Need a timeout/TTL to prevent leaks
   if Prefill never reads (e.g., request cancelled).

2. **Multi-image batching**: A request with N images may benefit from sending
   all N to the same Encode worker (better batching). The routing strategy
   should consider this vs distributing across workers.

3. **Encode worker failure**: If an Encode worker fails after returning a
   handle but before Prefill reads, the handle is stale. Need a retry path
   that re-encodes on a different worker.

4. **Embedding cache key propagation**: The Encode worker's cache uses
   `sha256(image_url)` as key. If the Frontend changes the URL format
   (e.g., adds auth tokens), cache keys won't match. Should the cache key
   be computed at the Frontend and passed through?
