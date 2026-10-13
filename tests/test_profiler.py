import json
import pytest
from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.profiler.benchmark import BaselineProfiler, BenchmarkResult, WindowMetric


class TestBaselineProfiler:
    def test_profiler_execution_and_schema(self, tmp_path):
        backbone = SveltoBackbone(model_name="mock", hidden_dim=64, num_layers=1)
        tokenizer = GenomicTokenizer()
        profiler = BaselineProfiler(
            backbone=backbone,
            tokenizer=tokenizer,
            lengths=[1024, 2048],
            num_warmup=1,
            num_repeats=2,
        )

        result = profiler.run()
        assert isinstance(result, BenchmarkResult)
        assert len(result.metrics) == 2

        # Check telemetry schema
        res_dict = result.to_dict()
        assert "timestamp" in res_dict
        assert "system" in res_dict
        assert "model_config" in res_dict
        assert "metrics" in res_dict

        assert res_dict["model_config"]["trainable_parameters"] == 0

        metric_1k = res_dict["metrics"][0]
        assert metric_1k["seq_length"] == 1024
        assert metric_1k["mean_latency_ms"] > 0
        assert metric_1k["peak_memory_mb"] > 0
        assert metric_1k["throughput_nt_per_sec"] > 0

        # Verify JSON export
        out_file = tmp_path / "telemetry.json"
        out_file.write_text(result.to_json())

        loaded = json.loads(out_file.read_text())
        assert loaded["metrics"][0]["seq_length"] == 1024


def test_profiler_tokenizes_with_the_backbones_own_vocabulary(monkeypatch):
    backbone = SveltoBackbone(model_name="mock", hidden_dim=32, num_layers=1)
    calls = []
    real_encode = backbone.encode
    monkeypatch.setattr(backbone, "encode", lambda s: calls.append(s) or real_encode(s))
    BaselineProfiler(backbone=backbone, lengths=[64], num_warmup=0, num_repeats=1).run()
    assert calls and len(calls[0]) == 64
