import { describe, expect, it } from "vitest";
import { syntheticRegion } from "@/lib/splice/sequence";
import { LocalHeuristicClient, WebSocketClient, type WireResponse } from "@/lib/splice/client";

const region = syntheticRegion();
const F = region.feature;

describe("LocalHeuristicClient", () => {
  it("returns reference and mutant tracks for the requested window", async () => {
    const c = new LocalHeuristicClient(region);
    const r = await c.mutate({ position: F + 1, ref: "T", alt: "A", windowStart: F - 100, windowEnd: F + 100 });
    expect(r.scores.start).toBe(F - 100);
    expect(r.scores.refDonor.length).toBe(200);
    expect(r.scores.refDonor[100]).toBeCloseTo(1, 2);
    expect(r.scores.mutDonor[100]).toBeCloseTo(0, 2);
    expect(r.telemetry?.source).toBe("simulated");
    expect(c.engine).toBe("Heuristic scorer");
  });
});

class FakeSocket {
  static last: FakeSocket;
  sent: string[] = [];
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) { FakeSocket.last = this; queueMicrotask(() => this.onopen?.()); }
  send(s: string) { this.sent.push(s); }
  close() { this.onclose?.(); }
  reply(r: WireResponse) { this.onmessage?.({ data: JSON.stringify(r) }); }
}

const wire = (requestId: string, over: Partial<WireResponse> = {}): WireResponse => ({
  session_id: "s", request_id: requestId, status: "success", engine: "hyenadna-small+heads",
  window_start: 10, window_end: 12,
  p_ref: [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], p_mut: [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]],
  delta_scores: { donor_gain: [0, 0], donor_loss: [0.75, 0], acceptor_gain: [0, 0], acceptor_loss: [0, 0] },
  latency_ms: 42.5,
  telemetry: { vanilla_latency_ms: 118.2, speedup_ratio: 2.78, acceptance_rate: 0.78, active_flops_saved: 0.64, source: "measured" },
  ...over,
});

describe("WebSocketClient (#6 contract)", () => {
  it("sends the #6 mutate payload and maps the response", async () => {
    const c = new WebSocketClient("ws://x/ws/splice-session", "s", FakeSocket as unknown as typeof WebSocket);
    const p = c.mutate({ position: 11, ref: "G", alt: "A", windowStart: 10, windowEnd: 12 });
    await new Promise((r) => setTimeout(r, 0));
    const msg = JSON.parse(FakeSocket.last.sent[0]);
    expect(msg).toMatchObject({ session_id: "s", action: "mutate", locus_position: 11, ref_base: "G", mut_base: "A", window_start: 10, window_end: 12 });
    FakeSocket.last.reply(wire(msg.request_id));
    const r = await p;
    expect(Array.from(r.scores.refDonor)).toEqual([0.8, 0.05].map(Math.fround));
    expect(r.latencyMs).toBe(42.5);
    expect(r.telemetry).toMatchObject({ speedup: 2.78, source: "measured" });
    expect(c.engine).toBe("hyenadna-small+heads");
  });
  it("rejects a request the server cancelled as stale", async () => {
    const c = new WebSocketClient("ws://x", "s", FakeSocket as unknown as typeof WebSocket);
    const p = c.mutate({ position: 11, ref: "G", alt: "A", windowStart: 10, windowEnd: 12 });
    await new Promise((r) => setTimeout(r, 0));
    const id = JSON.parse(FakeSocket.last.sent[0]).request_id;
    FakeSocket.last.reply(wire(id, { status: "cancelled" }));
    await expect(p).rejects.toThrow(/cancelled/);
  });
});
