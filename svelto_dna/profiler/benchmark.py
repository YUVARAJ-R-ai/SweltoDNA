"""
Baseline inference profiling module for genomic foundation backbones.
Measures wall-clock latency, peak memory (GPU VRAM / CPU RSS), and inference throughput.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import gc
import json
import os
import platform
import random
import time
import numpy as np
import psutil
import torch

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.core.tokenizer import GenomicTokenizer


@dataclass
class WindowMetric:
    """Metrics recorded for a single sequence length window."""
    seq_length: int
    mean_latency_ms: float
    std_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    peak_memory_mb: float
    throughput_nt_per_sec: float
    iterations: int


@dataclass
class BenchmarkResult:
    """Full benchmark telemetry output."""
    timestamp: str
    system: Dict[str, Any]
    model_config: Dict[str, Any]
    metrics: List[WindowMetric]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BaselineProfiler:
    """
    Automated profiler measuring latency, memory footprint, and throughput
    across varying genomic sequence lengths L in {1024, 2048, 5000, 10000}.
    """

    def __init__(
        self,
        backbone: SveltoBackbone,
        tokenizer: Optional[GenomicTokenizer] = None,
        lengths: Sequence[int] = (1024, 2048, 5000, 10000),
        num_warmup: int = 5,
        num_repeats: int = 20,
    ) -> None:
        self.backbone = backbone
        self.tokenizer = tokenizer or GenomicTokenizer()
        self.lengths = list(lengths)
        self.num_warmup = num_warmup
        self.num_repeats = num_repeats
        self.device = backbone.target_device
        self.is_cuda = self.device.type == "cuda"

    def _generate_synthetic_dna(self, length: int) -> str:
        """Generates random genomic sequence with standard A, C, G, T distribution."""
        random.seed(42 + length)
        bases = ["A", "C", "G", "T"]
        return "".join(random.choices(bases, k=length))

    def _get_peak_memory_mb(self) -> float:
        """Retrieves peak GPU VRAM if CUDA, else current process RSS memory in MB."""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024.0 * 1024.0)

    def _reset_peak_memory(self) -> None:
        """Resets CUDA peak memory statistics and triggers garbage collection."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def profile_window(self, seq_len: int) -> WindowMetric:
        """Profiles a single context window length."""
        raw_seq = self._generate_synthetic_dna(seq_len)
        encoded = self.backbone.encode(raw_seq)   # the backbone's own vocabulary (real checkpoints differ from GenomicTokenizer)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        self._reset_peak_memory()

        # Warmup iterations
        for _ in range(self.num_warmup):
            _ = self.backbone(input_ids, attention_mask=attention_mask)
            if self.is_cuda:
                torch.cuda.synchronize()

        self._reset_peak_memory()
        latencies_sec: List[float] = []

        # Measurement iterations
        for _ in range(self.num_repeats):
            if self.is_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = self.backbone(input_ids, attention_mask=attention_mask)

            if self.is_cuda:
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latencies_sec.append(end_time - start_time)

        peak_mem_mb = self._get_peak_memory_mb()
        latencies_ms = np.array(latencies_sec) * 1000.0

        mean_lat_ms = float(np.mean(latencies_ms))
        std_lat_ms = float(np.std(latencies_ms))
        med_lat_ms = float(np.median(latencies_ms))
        p95_lat_ms = float(np.percentile(latencies_ms, 95))
        min_lat_ms = float(np.min(latencies_ms))
        max_lat_ms = float(np.max(latencies_ms))

        mean_lat_sec = float(np.mean(latencies_sec))
        throughput_nt_sec = float(seq_len / mean_lat_sec) if mean_lat_sec > 0 else 0.0

        return WindowMetric(
            seq_length=seq_len,
            mean_latency_ms=round(mean_lat_ms, 3),
            std_latency_ms=round(std_lat_ms, 3),
            median_latency_ms=round(med_lat_ms, 3),
            p95_latency_ms=round(p95_lat_ms, 3),
            min_latency_ms=round(min_lat_ms, 3),
            max_latency_ms=round(max_lat_ms, 3),
            peak_memory_mb=round(peak_mem_mb, 2),
            throughput_nt_per_sec=round(throughput_nt_sec, 1),
            iterations=self.num_repeats,
        )

    def run(self) -> BenchmarkResult:
        """Executes full benchmark suite across all configured lengths."""
        metrics: List[WindowMetric] = []
        for length in self.lengths:
            metric = self.profile_window(length)
            metrics.append(metric)

        result = BenchmarkResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            system={
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "device": str(self.device),
                "is_cuda": self.is_cuda,
                "memory_metric_type": "cuda_vram_mb" if self.is_cuda else "cpu_rss_mb",
            },
            model_config={
                "model_name": self.backbone.model_name,
                "is_mock": self.backbone.is_mock,
                "trainable_parameters": self.backbone.trainable_parameters_count,
                "total_parameters": self.backbone.total_parameters_count,
                "target_dtype": str(self.backbone.target_dtype),
            },
            metrics=metrics,
        )
        return result
