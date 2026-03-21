"""System metrics collector using psutil and powermetrics."""

import asyncio
import logging
import platform
import re
import subprocess
from typing import Optional, Dict, Any

import psutil

from .prometheus_metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Background collector for system metrics."""

    def __init__(
        self,
        metrics: PrometheusMetrics,
        interval: float = 2.0,
        gpu_enabled: bool = True,
    ):
        self.metrics = metrics
        self.interval = interval
        self.gpu_enabled = gpu_enabled and metrics.is_apple_silicon()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._retry_count = 0
        self._max_retries = 3

    async def start(self):
        """Start the background collection task."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("System metrics collector started")

    async def stop(self):
        """Stop the background collection task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics collector stopped")

    async def _collect_loop(self):
        """Main collection loop with retry logic."""
        while self._running:
            try:
                await self._collect_once()
                self._retry_count = 0
            except Exception as e:
                logger.warning(f"Metrics collection error: {e}")
                self._retry_count += 1
                if self._retry_count >= self._max_retries:
                    logger.error("Max retries reached, disabling GPU metrics")
                    self.gpu_enabled = False
                    self._retry_count = 0
                else:
                    # Exponential backoff
                    await asyncio.sleep(self.interval * (2**self._retry_count))
                    continue

            await asyncio.sleep(self.interval)

    async def _collect_once(self):
        """Collect all metrics once."""
        # CPU and memory (always available)
        self._collect_cpu_memory()

        # GPU (Apple Silicon only)
        if self.gpu_enabled:
            await self._collect_gpu_metrics()

    def _collect_cpu_memory(self):
        """Collect CPU and memory metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.cpu_percent.set(cpu_percent)

            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_rss.set(memory_info.rss)
        except Exception as e:
            logger.warning(f"CPU/memory collection failed: {e}")

    async def _collect_gpu_metrics(self):
        """Collect GPU metrics using powermetrics."""
        try:
            # Run powermetrics with timeout
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "powermetrics",
                "--samplers",
                "gpu_power",
                "-i",
                "2000",
                "-n",
                "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("powermetrics timed out")
                return

            if proc.returncode != 0:
                error_msg = stderr.decode().strip()
                if "permission" in error_msg.lower():
                    logger.warning(
                        "powermetrics permission denied. "
                        "Run scripts/setup-gpu-metrics.sh"
                    )
                    self.gpu_enabled = False
                else:
                    logger.warning(f"powermetrics failed: {error_msg}")
                return

            # Parse output
            self._parse_powermetrics(stdout.decode())

        except FileNotFoundError:
            logger.warning("powermetrics not found, disabling GPU metrics")
            self.gpu_enabled = False
        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}")

    def _parse_powermetrics(self, output: str):
        """Parse powermetrics output and update gauges."""
        # Example output:
        # GPU Power: 2345 mW
        # GPU Active Residency: 78%
        # GPU Frequency: 1000 MHz

        # Active residency
        match = re.search(r"GPU Active Residency:\s*([\d.]+)%", output)
        if match:
            self.metrics.gpu_active.set(float(match.group(1)))

        # Frequency
        match = re.search(r"GPU Frequency:\s*([\d.]+)\s*MHz", output)
        if match:
            self.metrics.gpu_freq.set(float(match.group(1)))

        # Temperature (if available)
        match = re.search(r"GPU Temperature:\s*([\d.]+)", output)
        if match:
            self.metrics.gpu_temp.set(float(match.group(1)))
