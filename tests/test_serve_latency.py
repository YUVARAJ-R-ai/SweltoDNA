"""
Issue #6 AC: WebSocket round trip (edit sent → scores received) under 100 ms for 1 kb windows,
with the real HyenaDNA backbone on GPU. Latency does not depend on how well the head is trained.
"""

import statistics
import time

import pytest
import torch

pytest.importorskip("fastapi", reason="install the 'serve' extra")
from fastapi.testclient import TestClient  # noqa: E402

from svelto_dna.core.backbone import SveltoBackbone  # noqa: E402
from svelto_dna.model.splice_head import SpliceHead, SplicePredictor  # noqa: E402
from svelto_dna.serve.app import create_app  # noqa: E402
from svelto_dna.serve.regions import synthetic_region  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="latency target is for GPU")
def test_round_trip_under_100ms_for_1kb_window():
    try:
        bb = SveltoBackbone(model_name="LongSafari/hyenadna-small-32k-seqlen-hf")
    except RuntimeError:
        pytest.skip("HyenaDNA checkpoint not reachable")
    pred = SplicePredictor(bb, SpliceHead(hidden_dim=bb.hidden_dim))
    pred.engine = "hyenadna-small-32k + head"
    region = synthetic_region()
    F = region.feature
    times = []
    with TestClient(create_app(pred, region, context=1000)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        for k in range(25):
            pos = F - 200 + k * 16
            alt = "C" if region.seq[pos] != "C" else "G"
            t0 = time.perf_counter()
            ws.send_json({"session_id": "s", "request_id": str(k), "action": "mutate", "locus_position": pos,
                          "ref_base": region.seq[pos], "mut_base": alt, "window_start": F - 500, "window_end": F + 500})
            r = ws.receive_json()
            times.append((time.perf_counter() - t0) * 1000)
            assert r["status"] == "success"
    steady = times[3:]                                   # first calls include CUDA warm-up
    print(f"\nround trip 1 kb: median {statistics.median(steady):.1f} ms, p95 {sorted(steady)[int(len(steady) * 0.95) - 1]:.1f} ms")
    assert statistics.median(steady) < 100
    assert max(steady) < 100
