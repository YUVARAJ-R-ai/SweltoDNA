"""
FastAPI app: /ws/splice-session (contract in docs/superpowers/specs/2026-10-09-splice-explorer-frontend-design.md).

Cancellation: edits apply to session state immediately (cheap, never lost). Scoring runs on one GPU
worker thread. A queued request is skipped if a newer one arrived for the same session (stale work is
avoided, not just hidden); a request already running finishes but is reported as "cancelled".
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from svelto_dna.serve.regions import Region
from svelto_dna.serve.session import SpliceSession
from svelto_dna.speculative.verify import SpeculativeISM


def create_app(predictor, region: Region, context: int = 1000) -> FastAPI:
    app = FastAPI(title="Svelto-DNA splice server")
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="svelto-gpu")
    engine = getattr(predictor, "engine", type(predictor).__name__)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "engine": engine, "region": {k: v for k, v in region.to_wire().items() if k != "sequence"}}

    @app.websocket("/ws/splice-session")
    async def splice_session(ws: WebSocket) -> None:
        await ws.accept()
        session = SpliceSession(predictor, region, context)
        send_lock = asyncio.Lock()
        latest = {"gen": 0}
        tasks: set = set()

        async def send(msg: Dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_json(msg)

        def base(msg: Dict[str, Any], status: str) -> Dict[str, Any]:
            return {"session_id": msg.get("session_id"), "request_id": msg.get("request_id"), "status": status, "engine": engine}

        def scan(msg: Dict[str, Any]) -> Dict[str, Any]:
            """Speculative ISM on the reference (#4). Vanilla time = (1 + candidates) × this scan's measured single pass."""
            ism = SpeculativeISM(predictor, context=context, delta_window=50, threshold=float(msg.get("threshold", 0.2)))
            seq = "".join(region.seq)
            start, end = max(0, int(msg["window_start"])), min(len(seq), int(msg["window_end"]))
            res = ism.speculative(seq, start, end, k=int(msg.get("k", 48)))
            n = len(res.candidates)
            vanilla = (1 + n) * res.timings_ms["reference"]
            return {
                "type": "scan", "window_start": start, "window_end": end, "verified": res.verified,
                "candidates": [[p, b, None if np.isnan(d) else round(float(d), 4)] for (p, b), d in zip(res.candidates, res.deltas)],
                "latency_ms": round(res.timings_ms["total"], 2),
                "telemetry": {"vanilla_latency_ms": round(vanilla, 2), "speedup_ratio": round(vanilla / res.timings_ms["total"], 2),
                              "acceptance_rate": res.acceptance_rate, "active_flops_saved": 1 - (1 + res.verified) / (1 + n), "source": "measured",
                              "basis": f"vanilla = (1 + {n} candidates) x measured single-pass latency ({res.timings_ms['reference']:.1f} ms) of this scan"},
            }

        async def run(gen: int, msg: Dict[str, Any]) -> None:
            def work():
                if gen != latest["gen"]:
                    return None                                        # superseded before it started: skip the GPU work
                if msg.get("action") == "scan":
                    return scan(msg)
                t0 = time.perf_counter()
                out = session.score(int(msg.get("window_start", 0)), int(msg.get("window_end", len(region.seq))))
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                return out
            try:
                out = await asyncio.get_running_loop().run_in_executor(gpu, work)
                if out is None or gen != latest["gen"]:
                    await send({**base(msg, "cancelled")})
                else:
                    await send({**base(msg, "success"), "telemetry": None, **out})
            except Exception as e:                                     # report, keep the session alive
                await send({**base(msg, "error"), "error": str(e)})

        await send({"type": "hello", "engine": engine, "region": region.to_wire()})
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "ping":
                    await send({"type": "pong", "request_id": msg.get("request_id")})
                    continue
                try:
                    if action == "mutate":
                        session.apply_edit(int(msg["locus_position"]), msg["ref_base"], msg["mut_base"])
                    elif action == "reset":
                        session.reset()
                        await send({**base(msg, "success"), "window_start": 0, "window_end": 0, "p_ref": [], "p_mut": [],
                                    "delta_scores": {}, "latency_ms": 0, "telemetry": None})
                        continue
                    elif action not in ("score", "scan"):
                        raise ValueError(f"unknown action {action!r}")
                except (KeyError, ValueError) as e:
                    await send({**base(msg, "error"), "error": str(e)})
                    continue
                latest["gen"] += 1
                t = asyncio.create_task(run(latest["gen"], msg))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()

    return app
