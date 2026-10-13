"""
Issue #6: WebSocket scoring server (/ws/splice-session).
Uses a fast deterministic fake predictor; model quality is tested elsewhere.
"""

import hashlib
import threading
import time

import pytest
import torch

pytest.importorskip("fastapi", reason="install the 'serve' extra")
from fastapi.testclient import TestClient  # noqa: E402

from svelto_dna.serve.app import create_app  # noqa: E402
from svelto_dna.serve.regions import synthetic_region  # noqa: E402

FRONTEND_SYNTHETIC_SHA256 = "75c91572cf4334b435b24d7af2db9b0689c03d1f35c2fb1f309acd57431c158d"   # web/lib/splice/sequence.ts


class FakePredictor:
    """Donor probability 0.9 wherever the sequence reads GT, else ~0. Optional delay to simulate GPU work."""
    engine = "fake-gt-detector"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()

    def predict(self, seqs):
        with self.lock:
            self.calls += 1
        time.sleep(self.delay)
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, 3)
        out[..., 0] = 1.0
        for i, s in enumerate(seqs):
            for j in range(len(s) - 1):
                if s[j:j + 2] == "GT":
                    out[i, j] = torch.tensor([0.1, 0.9, 0.0])
        return out


def _mutate(ws, rid, pos, ref, alt, start, end):
    ws.send_json({"session_id": "s", "request_id": rid, "action": "mutate", "locus_position": pos, "ref_base": ref, "mut_base": alt,
                  "window_start": start, "window_end": end})


def test_synthetic_region_matches_the_frontend_byte_for_byte():
    r = synthetic_region()
    assert hashlib.sha256("".join(r.seq).encode()).hexdigest() == FRONTEND_SYNTHETIC_SHA256


def test_hello_ping_and_health():
    app = create_app(FakePredictor(), synthetic_region())
    with TestClient(app) as c:
        assert c.get("/health").json()["engine"] == "fake-gt-detector"
        with c.websocket_connect("/ws/splice-session") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["engine"] == "fake-gt-detector" and hello["region"]["length"] == 10000
            ws.send_json({"session_id": "s", "request_id": "p1", "action": "ping"})
            assert ws.receive_json() == {"type": "pong", "request_id": "p1"}


def test_mutate_returns_the_spec_contract_with_ref_and_mut_tracks():
    region = synthetic_region()
    F = region.feature
    with TestClient(create_app(FakePredictor(), region)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        _mutate(ws, "r1", F + 1, "T", "A", F - 10, F + 10)
        r = ws.receive_json()
        assert r["status"] == "success" and r["request_id"] == "r1" and r["engine"] == "fake-gt-detector"
        assert (r["window_start"], r["window_end"]) == (F - 10, F + 10)
        assert len(r["p_ref"]) == len(r["p_mut"]) == 20 and len(r["p_ref"][0]) == 3
        assert r["p_ref"][10][1] == pytest.approx(0.9) and r["p_mut"][10][1] == pytest.approx(0.0)     # GT destroyed
        assert r["delta_scores"]["donor_loss"][10] == pytest.approx(0.9)
        assert r["latency_ms"] >= 0 and r["telemetry"] is None


def test_score_reflects_session_edits_and_reset_clears_them():
    region = synthetic_region()
    F = region.feature
    with TestClient(create_app(FakePredictor(), region)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        _mutate(ws, "m", F + 1, "T", "A", F - 5, F + 5); ws.receive_json()
        ws.send_json({"session_id": "s", "request_id": "s1", "action": "score", "window_start": F - 5, "window_end": F + 5})
        assert ws.receive_json()["p_mut"][5][1] == pytest.approx(0.0)
        ws.send_json({"session_id": "s", "request_id": "x", "action": "reset"}); ws.receive_json()
        ws.send_json({"session_id": "s", "request_id": "s2", "action": "score", "window_start": F - 5, "window_end": F + 5})
        assert ws.receive_json()["p_mut"][5][1] == pytest.approx(0.9)


def test_wrong_reference_base_is_an_error_not_a_silent_edit():
    region = synthetic_region()
    with TestClient(create_app(FakePredictor(), region)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        _mutate(ws, "bad", region.feature + 1, "G", "A", 0, 10)            # reference there is T
        r = ws.receive_json()
        assert r["status"] == "error" and "reference" in r["error"]


def test_rapid_edits_cancel_stale_requests_and_only_the_latest_is_scored():
    """#6 AC: 10 clicks in 500 ms. Stale requests come back 'cancelled'; the last succeeds; no crash."""
    region = synthetic_region()
    F = region.feature
    pred = FakePredictor(delay=0.12)
    with TestClient(create_app(pred, region)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        for k in range(10):
            _mutate(ws, f"r{k}", F + 2 + k, region.seq[F + 2 + k], "C" if region.seq[F + 2 + k] != "C" else "G", F - 20, F + 20)
            time.sleep(0.05)
        replies = {}
        while len(replies) < 10:
            r = ws.receive_json()
            replies[r["request_id"]] = r["status"]
        assert replies["r9"] == "success"
        assert sum(s == "cancelled" for s in replies.values()) >= 5
        assert pred.calls < 10                     # stale work was skipped, not just hidden


def test_concurrent_clients_are_isolated_and_deterministic():
    region = synthetic_region()
    F = region.feature
    with TestClient(create_app(FakePredictor(), region)) as c:
        with c.websocket_connect("/ws/splice-session") as a, c.websocket_connect("/ws/splice-session") as b:
            a.receive_json(); b.receive_json()
            _mutate(a, "a1", F + 1, "T", "A", F - 5, F + 5)
            b.send_json({"session_id": "b", "request_id": "b1", "action": "score", "window_start": F - 5, "window_end": F + 5})
            ra, rb = a.receive_json(), b.receive_json()
            assert ra["p_mut"][5][1] == pytest.approx(0.0)          # a's edit applied in a
            assert rb["p_mut"][5][1] == pytest.approx(0.9)          # b never sees it
            _mutate(b, "b2", F + 1, "T", "A", F - 5, F + 5)
            assert b.receive_json()["p_mut"] == ra["p_mut"]          # same edit, same answer


def test_scan_runs_speculative_ism_and_reports_measured_telemetry():
    """#4 + #9: the server's scan returns verified deltas and measured (not simulated) telemetry."""
    region = synthetic_region()
    F = region.feature
    with TestClient(create_app(FakePredictor(), region, context=200)) as c, c.websocket_connect("/ws/splice-session") as ws:
        ws.receive_json()
        ws.send_json({"session_id": "s", "request_id": "scan1", "action": "scan", "window_start": F - 8, "window_end": F + 8, "k": 12})
        r = ws.receive_json()
        assert r["status"] == "success" and r["type"] == "scan"
        assert len(r["candidates"]) == 48 and r["verified"] == 12
        verified = [cand for cand in r["candidates"] if cand[2] is not None]
        assert len(verified) == 12 and all(cand[0] - (F - 8) in range(16) for cand in verified)
        t = r["telemetry"]
        assert t["source"] == "measured" and "basis" in t
        assert t["active_flops_saved"] == pytest.approx(1 - 13 / 49)
        assert 0 <= t["acceptance_rate"] <= 1 and t["speedup_ratio"] > 0
