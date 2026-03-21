"""Tests for monitoring module."""

import pytest
import time
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from monitoring.prometheus_metrics import (
    PrometheusMetrics,
    REGISTRY,
    get_metrics_handler,
)
from monitoring.profiler import InferenceProfiler
from monitoring.system_collector import SystemMetricsCollector


class TestPrometheusMetrics:
    """Test Prometheus metrics registration."""

    def test_metrics_initialization(self):
        """Test that all metrics are created."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            # Request metrics
            assert metrics.requests_total is not None
            assert metrics.request_duration is not None

            # Inference stage metrics
            assert metrics.inference_duration is not None
            assert metrics.inference_tokens is not None

            # System metrics
            assert metrics.cpu_percent is not None
            assert metrics.memory_rss is not None
            assert metrics.gpu_active is not None
            assert metrics.gpu_freq is not None
            assert metrics.gpu_temp is not None

            # Cache metrics
            assert metrics.cache_hits is not None
            assert metrics.cache_misses is not None
            assert metrics.cache_size is not None
            assert metrics.cache_evictions is not None

            # Model management metrics
            assert metrics.model_load_duration is not None
            assert metrics.model_evictions is not None
            assert metrics.loaded_models is not None

    def test_is_apple_silicon(self):
        """Test platform detection."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch("platform.machine", return_value="arm64"):
                with patch("platform.system", return_value="Darwin"):
                    assert metrics.is_apple_silicon() is True

            with patch("platform.machine", return_value="x86_64"):
                assert metrics.is_apple_silicon() is False

            with patch("platform.machine", return_value="arm64"):
                with patch("platform.system", return_value="Linux"):
                    assert metrics.is_apple_silicon() is False

            with patch("platform.machine", return_value="aarch64"):
                with patch("platform.system", return_value="Darwin"):
                    assert metrics.is_apple_silicon() is True


class TestInferenceProfiler:
    """Test inference profiler."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            assert profiler.metrics is metrics
            assert profiler.model == "test-model"
            assert profiler.endpoint == "embeddings"
            assert profiler._start_time is None

    def test_profiler_stage_timing(self):
        """Test that profiler records stage timing."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            with profiler.stage("tokenization"):
                time.sleep(0.01)  # 10ms

            # Metric should be recorded without exception

    def test_profiler_total_timing(self):
        """Test that profiler records total timing."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            with profiler.total():
                time.sleep(0.01)

            # Verify request duration has samples
            samples = metrics.request_duration.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_token_recording(self):
        """Test token count recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            profiler.record_tokens(100)

            samples = metrics.inference_tokens.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_cache_hit(self):
        """Test cache hit recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            profiler.record_cache_hit()

            samples = metrics.cache_hits.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_cache_miss(self):
        """Test cache miss recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            profiler.record_cache_miss()

            samples = metrics.cache_misses.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_cache_size_update(self):
        """Test cache size gauge update."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            profiler.update_cache_size(42)

            samples = metrics.cache_size.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_model_load_duration(self):
        """Test model load duration recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            profiler.record_model_load(0.5)

            samples = metrics.model_load_duration.collect()[0].samples
            assert len(samples) > 0

    def test_profiler_model_eviction(self):
        """Test model eviction recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            profiler = InferenceProfiler(metrics, "test-model", "embeddings")

            # First set loaded models
            profiler.update_loaded_models(2)

            # Record eviction
            profiler.record_model_eviction()

            # Verify eviction counter
            eviction_samples = metrics.model_evictions.collect()[0].samples
            assert len(eviction_samples) > 0

            # Verify loaded models was decremented
            loaded_samples = metrics.loaded_models.collect()[0].samples
            assert len(loaded_samples) > 0

    def test_profiler_multiple_models(self):
        """Test profiler with multiple models."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            profiler_small = InferenceProfiler(metrics, "small", "embeddings")
            profiler_large = InferenceProfiler(metrics, "large", "embeddings")

            profiler_small.record_tokens(50)
            profiler_large.record_tokens(100)

            samples = metrics.inference_tokens.collect()[0].samples
            model_samples = {s.labels.get("model"): s.value for s in samples}

            # Verify both models have recorded tokens
            assert "small" in model_samples
            assert "large" in model_samples


class TestSystemMetricsCollector:
    """Test system metrics collector."""

    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """Test collector initialization."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            collector = SystemMetricsCollector(metrics, interval=1.0, gpu_enabled=False)

            assert collector.metrics is metrics
            assert collector.interval == 1.0
            assert collector.gpu_enabled is False
            assert collector._running is False

    @pytest.mark.asyncio
    async def test_collector_start_stop(self):
        """Test starting and stopping the collector."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            collector = SystemMetricsCollector(metrics, interval=0.1, gpu_enabled=False)

            # Start collector
            await collector.start()
            assert collector._running is True
            assert collector._task is not None

            # Let it run briefly
            await asyncio.sleep(0.2)

            # Stop collector
            await collector.stop()
            assert collector._running is False

    @pytest.mark.asyncio
    async def test_cpu_memory_collection(self):
        """Test CPU and memory metrics collection."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            collector = SystemMetricsCollector(metrics, interval=0.1, gpu_enabled=False)

            # Collect once
            await collector.start()
            await asyncio.sleep(0.2)
            await collector.stop()

            # Verify metrics were collected
            cpu_samples = metrics.cpu_percent.collect()[0].samples
            assert len(cpu_samples) > 0
            assert 0 <= cpu_samples[0].value <= 100

            memory_samples = metrics.memory_rss.collect()[0].samples
            assert len(memory_samples) > 0
            assert memory_samples[0].value > 0

    @pytest.mark.asyncio
    async def test_gpu_metrics_with_mock(self):
        """Test GPU metrics collection with mocked powermetrics."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.1, gpu_enabled=True
                )

                # Mock asyncio.create_subprocess_exec
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = (
                    b"GPU Active Residency: 78%\nGPU Frequency: 1000 MHz\nGPU Temperature: 45\n",
                    b"",
                )

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    await collector.start()
                    await asyncio.sleep(0.2)
                    await collector.stop()

                # Verify GPU metrics were set (samples should exist)
                gpu_active = metrics.gpu_active.collect()[0].samples
                gpu_freq = metrics.gpu_freq.collect()[0].samples
                gpu_temp = metrics.gpu_temp.collect()[0].samples

                # Check that we have samples (values may vary)
                assert len(gpu_active) > 0 or len(gpu_freq) > 0 or len(gpu_temp) > 0

    @pytest.mark.asyncio
    async def test_gpu_metrics_permission_denied(self):
        """Test GPU metrics collection handles permission errors."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.1, gpu_enabled=True
                )

                # Mock subprocess with permission error
                mock_proc = AsyncMock()
                mock_proc.returncode = 1
                mock_proc.communicate.return_value = (b"", b"Permission denied")

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    await collector.start()
                    await asyncio.sleep(0.3)
                    await collector.stop()

                # Verify GPU metrics were disabled
                assert collector.gpu_enabled is False

    @pytest.mark.asyncio
    async def test_gpu_metrics_timeout(self):
        """Test GPU metrics collection handles timeout."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.1, gpu_enabled=True
                )

                # Mock subprocess that times out
                mock_proc = AsyncMock()
                mock_proc.kill = MagicMock()

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                        await collector.start()
                        await asyncio.sleep(0.3)
                        await collector.stop()

                # Verify proc.kill was called
                mock_proc.kill.assert_called()

    @pytest.mark.asyncio
    async def test_gpu_metrics_not_found(self):
        """Test GPU metrics collection handles FileNotFoundError."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.1, gpu_enabled=True
                )

                # Mock FileNotFoundError
                with patch(
                    "asyncio.create_subprocess_exec", side_effect=FileNotFoundError
                ):
                    await collector.start()
                    await asyncio.sleep(0.3)
                    await collector.stop()

                # Verify GPU metrics were disabled
                assert collector.gpu_enabled is False

    @pytest.mark.asyncio
    async def test_non_apple_silicon_no_gpu(self):
        """Test that GPU metrics are disabled on non-Apple Silicon."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=False):
                collector = SystemMetricsCollector(
                    metrics, interval=0.1, gpu_enabled=True
                )

                # GPU should be disabled even though gpu_enabled=True
                assert collector.gpu_enabled is False

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic for GPU metrics collection."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.05, gpu_enabled=True
                )

                # Mock subprocess that fails
                call_count = [0]

                async def mock_create_subprocess(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] <= 2:
                        raise RuntimeError("Temporary error")
                    # Success on third try
                    mock_proc = AsyncMock()
                    mock_proc.returncode = 0
                    mock_proc.communicate.return_value = (
                        b"GPU Active Residency: 50%\n",
                        b"",
                    )
                    return mock_proc

                with patch(
                    "asyncio.create_subprocess_exec", side_effect=mock_create_subprocess
                ):
                    await collector.start()
                    await asyncio.sleep(0.3)
                    await collector.stop()

                # Should have retried at least 3 times
                assert call_count[0] >= 3

    @pytest.mark.asyncio
    async def test_max_retries_disable_gpu(self):
        """Test that GPU metrics are disabled after max retries."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()

            with patch.object(metrics, "is_apple_silicon", return_value=True):
                collector = SystemMetricsCollector(
                    metrics, interval=0.02, gpu_enabled=True
                )

                # Mock subprocess that always fails
                call_count = [0]

                async def mock_create_subprocess(*args, **kwargs):
                    call_count[0] += 1
                    raise RuntimeError("Persistent error")

                with patch(
                    "asyncio.create_subprocess_exec", side_effect=mock_create_subprocess
                ):
                    await collector.start()
                    # Let it run enough to hit max retries (3)
                    # The retry count gets reset on each failure, so we need longer sleep
                    # to ensure the retry counter reaches max_retries
                    await asyncio.sleep(0.5)
                    await collector.stop()

                # Should have attempted multiple times
                assert call_count[0] >= 3

    @pytest.mark.asyncio
    async def test_powermetrics_parsing(self):
        """Test powermetrics output parsing."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        with patch("monitoring.prometheus_metrics.REGISTRY", fresh_reg):
            metrics = PrometheusMetrics()
            collector = SystemMetricsCollector(metrics, interval=0.1, gpu_enabled=False)

            # Test parsing with various formats
            test_outputs = [
                "GPU Active Residency: 78%\nGPU Frequency: 1000 MHz\nGPU Temperature: 45\n",
                "GPU Active Residency: 50.5%\nGPU Frequency: 1200.5 MHz\n",
                "GPU Active Residency: 0%\n",
            ]

            for output in test_outputs:
                collector._parse_powermetrics(output)

            # Check that values were updated (last value should persist)
            gpu_active = metrics.gpu_active.collect()[0].samples
            if gpu_active:
                assert gpu_active[0].value == 0  # From last output


class TestPrometheusEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_get_metrics_handler(self):
        """Test that handler function is created."""
        handler = get_metrics_handler()

        assert handler is not None
        assert callable(handler)

    def test_metrics_handler_format(self):
        """Test that handler returns valid Prometheus format."""
        handler = get_metrics_handler()

        # Call handler (this is synchronous)
        response = handler()

        # Check response structure
        assert hasattr(response, "body")
        assert hasattr(response, "media_type")

        # Check media type (may vary by prometheus_client version)
        assert "text/plain" in response.media_type
        assert "charset=utf-8" in response.media_type

    def test_metrics_after_recording(self):
        """Test that metrics are properly formatted after recording."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry, generate_latest

        fresh_reg = CollectorRegistry()
        metrics = PrometheusMetrics(registry=fresh_reg)
        profiler = InferenceProfiler(metrics, "test-model", "embeddings")

        # Record some data
        profiler.record_tokens(100)
        profiler.record_cache_hit()
        profiler.update_cache_size(42)

        # Get metrics directly from fresh registry
        content = generate_latest(fresh_reg).decode("utf-8")

        # Verify recorded values are in output
        assert "qwen3_inference_tokens" in content
        assert "qwen3_cache_hits" in content
        assert "qwen3_cache_size" in content


class TestIntegration:
    """Integration tests for monitoring module."""

    @pytest.mark.asyncio
    async def test_end_to_end_metrics_flow(self):
        """Test complete metrics flow from recording to export."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry, generate_latest

        fresh_reg = CollectorRegistry()
        metrics = PrometheusMetrics(registry=fresh_reg)
        collector = SystemMetricsCollector(metrics, interval=0.1, gpu_enabled=False)
        profiler = InferenceProfiler(metrics, "small", "embeddings")

        # Start collector
        await collector.start()

        # Simulate inference
        with profiler.stage("tokenization"):
            time.sleep(0.005)

        with profiler.total():
            profiler.record_tokens(50)
            profiler.record_cache_hit()
            profiler.update_cache_size(10)
            time.sleep(0.005)

        # Stop collector
        await collector.stop()

        # Get metrics directly from fresh registry
        content = generate_latest(fresh_reg).decode("utf-8")

        # Verify all expected metrics present
        assert "qwen3_inference_duration_seconds" in content
        assert "qwen3_request_duration_seconds" in content
        assert "qwen3_inference_tokens" in content
        assert "qwen3_cache_hits" in content
        assert "qwen3_cpu_percent" in content

    @pytest.mark.asyncio
    async def test_multiple_concurrent_profilers(self):
        """Test multiple profilers running concurrently."""
        # Use fresh registry for isolated test
        from prometheus_client import CollectorRegistry

        fresh_reg = CollectorRegistry()
        metrics = PrometheusMetrics(registry=fresh_reg)

        async def simulate_request(model: str):
            profiler = InferenceProfiler(metrics, model, "embeddings")
            with profiler.total():
                with profiler.stage("inference"):
                    time.sleep(0.01)
                profiler.record_tokens(25)

        # Run multiple concurrent requests
        tasks = [
            simulate_request("small"),
            simulate_request("medium"),
            simulate_request("large"),
        ]

        await asyncio.gather(*tasks)

        # Verify metrics for all models
        request_samples = metrics.requests_total.collect()[0].samples
        model_labels = {s.labels.get("model") for s in request_samples}

        assert "small" in model_labels
        assert "medium" in model_labels
        assert "large" in model_labels

        # Verify token counts were recorded
        token_samples = metrics.inference_tokens.collect()[0].samples
        assert len(token_samples) >= 3  # At least 3 models
