# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics for the multimodal vision encoder pipeline.

Provides latency histograms, counters, and gauges for:
- Encode request throughput and errors
- Per-stage latency breakdown (image load, preprocess, vision encode, transfer)
- Embedding size distribution
- Encoder-side and prefill-side embedding cache hit/miss rates
"""

import logging

from prometheus_client import Counter, Gauge, Histogram

from dynamo.prometheus_names import multimodal as metric_names

logger = logging.getLogger(__name__)

# Histogram buckets tuned for vision pipeline stages
_ENCODE_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)
_EMBEDDING_BYTES_BUCKETS = (
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    float("inf"),
)
_IMAGES_PER_REQUEST_BUCKETS = (1, 2, 3, 4, 5, 8, 10, 16, 32, float("inf"))


class MultimodalMetricsCollector:
    """Prometheus metrics collector for the multimodal vision encoder pipeline.

    Args:
        labels: Dict with keys like model_name, component_name, worker_type.
    """

    def __init__(self, labels: dict):
        self._labelnames = list(labels.keys())
        self._labelvalues = list(labels.values())

        # --- Request counters ---
        self.encode_requests_total = Counter(
            metric_names.ENCODE_REQUESTS_TOTAL,
            "Total number of multimodal encode requests processed",
            labelnames=self._labelnames,
        )
        self.images_encoded_total = Counter(
            metric_names.IMAGES_ENCODED_TOTAL,
            "Total number of images encoded across all requests",
            labelnames=self._labelnames,
        )
        self.encode_errors_total = Counter(
            metric_names.ENCODE_ERRORS_TOTAL,
            "Total number of encode requests that failed",
            labelnames=self._labelnames,
        )

        # --- Latency histograms ---
        self.encode_request_duration = Histogram(
            metric_names.ENCODE_REQUEST_DURATION_SECONDS,
            "End-to-end encode request latency in seconds",
            labelnames=self._labelnames,
            buckets=_ENCODE_LATENCY_BUCKETS,
        )
        self.image_load_duration = Histogram(
            metric_names.IMAGE_LOAD_SECONDS,
            "Image loading latency in seconds",
            labelnames=self._labelnames,
            buckets=_ENCODE_LATENCY_BUCKETS,
        )
        self.image_preprocess_duration = Histogram(
            metric_names.IMAGE_PREPROCESS_SECONDS,
            "Image preprocessing latency in seconds",
            labelnames=self._labelnames,
            buckets=_ENCODE_LATENCY_BUCKETS,
        )
        self.vision_encode_duration = Histogram(
            metric_names.VISION_ENCODE_SECONDS,
            "Vision encoder forward pass latency in seconds",
            labelnames=self._labelnames,
            buckets=_ENCODE_LATENCY_BUCKETS,
        )
        self.embedding_transfer_duration = Histogram(
            metric_names.EMBEDDING_TRANSFER_SECONDS,
            "Embedding transfer preparation latency in seconds",
            labelnames=self._labelnames,
            buckets=_ENCODE_LATENCY_BUCKETS,
        )

        # --- Size histograms ---
        self.embedding_bytes = Histogram(
            metric_names.EMBEDDING_BYTES,
            "Embedding tensor size in bytes per image",
            labelnames=self._labelnames,
            buckets=_EMBEDDING_BYTES_BUCKETS,
        )
        self.images_per_request = Histogram(
            metric_names.IMAGES_PER_REQUEST,
            "Number of images per encode request",
            labelnames=self._labelnames,
            buckets=_IMAGES_PER_REQUEST_BUCKETS,
        )

        # --- Encoder-side cache gauges ---
        self.encoder_cache_hit_rate = Gauge(
            metric_names.ENCODER_CACHE_HIT_RATE,
            "Embedding cache hit rate (0.0-1.0) on the encode worker",
            labelnames=self._labelnames,
            multiprocess_mode="livemax",
        )
        self.encoder_cache_hits_total = Counter(
            metric_names.ENCODER_CACHE_HITS_TOTAL,
            "Embedding cache hits total on the encode worker",
            labelnames=self._labelnames,
        )
        self.encoder_cache_misses_total = Counter(
            metric_names.ENCODER_CACHE_MISSES_TOTAL,
            "Embedding cache misses total on the encode worker",
            labelnames=self._labelnames,
        )
        self.encoder_cache_bytes = Gauge(
            metric_names.ENCODER_CACHE_BYTES,
            "Embedding cache current size in bytes",
            labelnames=self._labelnames,
            multiprocess_mode="livemax",
        )
        self.encoder_cache_entries = Gauge(
            metric_names.ENCODER_CACHE_ENTRIES,
            "Embedding cache entry count",
            labelnames=self._labelnames,
            multiprocess_mode="livemax",
        )

        # --- Prefill-side cache gauges ---
        self.prefill_cache_hit_rate = Gauge(
            metric_names.PREFILL_CACHE_HIT_RATE,
            "Prefill-side embedding cache hit rate (0.0-1.0)",
            labelnames=self._labelnames,
            multiprocess_mode="livemax",
        )
        self.prefill_cache_hits_total = Counter(
            metric_names.PREFILL_CACHE_HITS_TOTAL,
            "Prefill-side embedding cache hits total",
            labelnames=self._labelnames,
        )
        self.prefill_cache_misses_total = Counter(
            metric_names.PREFILL_CACHE_MISSES_TOTAL,
            "Prefill-side embedding cache misses total",
            labelnames=self._labelnames,
        )

        logger.info("MultimodalMetricsCollector initialized")

    # --- Convenience methods ---

    def record_encode_request(self, duration_s: float, num_images: int):
        """Record a completed encode request."""
        lv = self._labelvalues
        self.encode_requests_total.labels(*lv).inc()
        self.images_encoded_total.labels(*lv).inc(num_images)
        self.encode_request_duration.labels(*lv).observe(duration_s)
        self.images_per_request.labels(*lv).observe(num_images)

    def record_encode_error(self):
        """Record a failed encode request."""
        self.encode_errors_total.labels(*self._labelvalues).inc()

    def record_image_load(self, duration_s: float):
        """Record image loading latency."""
        self.image_load_duration.labels(*self._labelvalues).observe(duration_s)

    def record_image_preprocess(self, duration_s: float):
        """Record image preprocessing latency."""
        self.image_preprocess_duration.labels(*self._labelvalues).observe(duration_s)

    def record_vision_encode(self, duration_s: float):
        """Record vision encoder forward pass latency."""
        self.vision_encode_duration.labels(*self._labelvalues).observe(duration_s)

    def record_embedding_transfer(self, duration_s: float):
        """Record embedding transfer preparation latency."""
        self.embedding_transfer_duration.labels(*self._labelvalues).observe(duration_s)

    def record_embedding_size(self, size_bytes: int):
        """Record embedding tensor size in bytes."""
        self.embedding_bytes.labels(*self._labelvalues).observe(size_bytes)

    def record_encoder_cache_stats(self, cache_stats: dict):
        """Update encoder-side cache gauges from cache.stats dict."""
        lv = self._labelvalues
        self.encoder_cache_hit_rate.labels(*lv).set(cache_stats.get("hit_rate", 0.0))
        self.encoder_cache_bytes.labels(*lv).set(cache_stats.get("current_bytes", 0))
        self.encoder_cache_entries.labels(*lv).set(cache_stats.get("entries", 0))

    def record_encoder_cache_hit(self):
        """Record a single encoder cache hit."""
        self.encoder_cache_hits_total.labels(*self._labelvalues).inc()

    def record_encoder_cache_miss(self):
        """Record a single encoder cache miss."""
        self.encoder_cache_misses_total.labels(*self._labelvalues).inc()

    def record_prefill_cache_hit(self):
        """Record a single prefill-side cache hit."""
        self.prefill_cache_hits_total.labels(*self._labelvalues).inc()

    def record_prefill_cache_miss(self):
        """Record a single prefill-side cache miss."""
        self.prefill_cache_misses_total.labels(*self._labelvalues).inc()

    def update_prefill_cache_hit_rate(self, hit_rate: float):
        """Update prefill-side cache hit rate gauge."""
        self.prefill_cache_hit_rate.labels(*self._labelvalues).set(hit_rate)
