/**
 * Transport for splice scoring. The UI talks only to `SpliceClient`; the local heuristic and the
 * #6 WebSocket server are interchangeable behind it.
 */
import type { Base, Region, WireRegion } from "./sequence";
import { regionFromWire } from "./sequence";
import { SpliceModel } from "./scorer";

export interface WindowScores {
  start: number;
  end: number;
  refDonor: Float32Array; refAcceptor: Float32Array;
  mutDonor: Float32Array; mutAcceptor: Float32Array;
}

export interface Telemetry {
  vanillaLatencyMs: number;
  speedup: number;
  acceptanceRate: number;
  flopsSaved: number;
  source: "measured" | "simulated";
  /** How a measured figure was obtained (e.g. how the vanilla baseline time was derived). */
  basis?: string;
}

export interface ScanCandidate { position: number; alt: Base; delta: number | null }   // null = not verified
export interface ScanResult { candidates: ScanCandidate[]; verified: number; latencyMs: number; telemetry: Telemetry | null }

export interface MutateResult { scores: WindowScores; latencyMs: number; telemetry: Telemetry | null }
export interface MutateRequest { position: number; ref: Base; alt: Base; windowStart: number; windowEnd: number }

export interface SpliceClient {
  readonly engine: string;
  score(windowStart: number, windowEnd: number): Promise<WindowScores>;
  mutate(req: MutateRequest): Promise<MutateResult>;
  reset(): Promise<void>;
  close(): void;
  /** Server-side speculative saturation scan (#4); absent for the local heuristic. */
  scan?(windowStart: number, windowEnd: number, k: number): Promise<ScanResult>;
}

/* ───────── #6 wire format (spec: docs/superpowers/specs/2026-10-09-splice-explorer-frontend-design.md) ───────── */
export interface WireRequest {
  session_id: string; request_id: string;
  action: "mutate" | "score" | "reset" | "scan" | "ping";
  k?: number;
  locus_position?: number; ref_base?: Base; mut_base?: Base;
  window_start?: number; window_end?: number;
}
export interface WireResponse {
  session_id: string; request_id: string;
  status: "success" | "cancelled" | "error";
  engine: string;
  window_start: number; window_end: number;
  p_ref: number[][]; p_mut: number[][];                 // L×3: [neither, donor, acceptor]
  delta_scores: { donor_gain: number[]; donor_loss: number[]; acceptor_gain: number[]; acceptor_loss: number[] };
  latency_ms: number;
  telemetry?: { vanilla_latency_ms: number; speedup_ratio: number; acceptance_rate: number; active_flops_saved: number; source: "measured" | "simulated"; basis?: string } | null;
  error?: string;
}

export function scoresFromWire(r: WireResponse): WindowScores {
  const col = (rows: number[][], k: number) => Float32Array.from(rows, (row) => row[k]);
  return { start: r.window_start, end: r.window_end, refDonor: col(r.p_ref, 1), refAcceptor: col(r.p_ref, 2), mutDonor: col(r.p_mut, 1), mutAcceptor: col(r.p_mut, 2) };
}
export function telemetryFromWire(t: WireResponse["telemetry"]): Telemetry | null {
  return t ? { vanillaLatencyMs: t.vanilla_latency_ms, speedup: t.speedup_ratio, acceptanceRate: t.acceptance_rate, flopsSaved: t.active_flops_saved, source: t.source, basis: t.basis } : null;
}

/* ───────── local heuristic (default; no server needed) ───────── */
export class LocalHeuristicClient implements SpliceClient {
  readonly engine = "Heuristic scorer";
  readonly model: SpliceModel;
  constructor(region: Region) { this.model = new SpliceModel(region); }

  private window(start: number, end: number): WindowScores {
    const m = this.model, s = Math.max(0, start), e = Math.min(m.length, end);
    return { start: s, end: e, refDonor: m.refDonor.slice(s, e), refAcceptor: m.refAcceptor.slice(s, e), mutDonor: m.curDonor.slice(s, e), mutAcceptor: m.curAcceptor.slice(s, e) };
  }
  async score(start: number, end: number) { return this.window(start, end); }
  async mutate(req: MutateRequest): Promise<MutateResult> {
    const t0 = performance.now();
    this.model.set(req.position, req.alt);
    const latencyMs = performance.now() - t0;
    const vanilla = 165 + Math.random() * 35, ms = 40 + Math.random() * 20;
    return {
      scores: this.window(req.windowStart, req.windowEnd), latencyMs,
      telemetry: { vanillaLatencyMs: vanilla, speedup: vanilla / ms, acceptanceRate: 0.68 + Math.random() * 0.14, flopsSaved: 0.54 + Math.random() * 0.16, source: "simulated" },
    };
  }
  async reset() { this.model.reset(); }
  close() {}
}

/* ───────── #6 WebSocket server ───────── */
export class WebSocketClient implements SpliceClient {
  engine = "Server";
  private ws: WebSocket;
  private ready: Promise<void>;
  private seq = 0;
  private pending = new Map<string, { resolve: (r: WireResponse) => void; reject: (e: Error) => void }>();
  /** Resolves with the server's region and engine (#6 hello message). */
  readonly hello: Promise<{ engine: string; region: Region }>;
  private resolveHello!: (h: { engine: string; region: Region }) => void;

  constructor(url: string, private sessionId: string, Impl: typeof WebSocket = WebSocket) {
    this.hello = new Promise((res) => { this.resolveHello = res; });
    this.ws = new Impl(url);
    this.ready = new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = () => reject(new Error(`Could not connect to ${url}`));
    });
    this.ws.onmessage = (e: MessageEvent) => {
      const raw = JSON.parse(e.data);
      if (raw.type === "hello") { this.engine = raw.engine; this.resolveHello({ engine: raw.engine, region: regionFromWire(raw.region as WireRegion) }); return; }
      if (raw.type === "pong") return;
      const r = raw as WireResponse;
      const p = this.pending.get(r.request_id);
      if (!p) return;
      this.pending.delete(r.request_id);
      if (r.engine) this.engine = r.engine;
      if (r.status === "success") p.resolve(r);
      else p.reject(new Error(r.status === "cancelled" ? "Request cancelled as stale" : r.error || "Server error"));
    };
    this.ws.onclose = () => { for (const p of this.pending.values()) p.reject(new Error("Connection closed")); this.pending.clear(); };
  }

  private async send(req: Omit<WireRequest, "session_id" | "request_id">): Promise<WireResponse> {
    await this.ready;
    const request_id = `${this.sessionId}-${++this.seq}`;
    return new Promise((resolve, reject) => {
      this.pending.set(request_id, { resolve, reject });
      this.ws.send(JSON.stringify({ session_id: this.sessionId, request_id, ...req }));
    });
  }
  async score(start: number, end: number) { return scoresFromWire(await this.send({ action: "score", window_start: start, window_end: end })); }
  async mutate(req: MutateRequest): Promise<MutateResult> {
    const r = await this.send({ action: "mutate", locus_position: req.position, ref_base: req.ref, mut_base: req.alt, window_start: req.windowStart, window_end: req.windowEnd });
    return { scores: scoresFromWire(r), latencyMs: r.latency_ms, telemetry: telemetryFromWire(r.telemetry) };
  }
  async reset() { await this.send({ action: "reset" }); }
  async scan(windowStart: number, windowEnd: number, k: number): Promise<ScanResult> {
    const r = (await this.send({ action: "scan", window_start: windowStart, window_end: windowEnd, k })) as unknown as
      { candidates: [number, Base, number | null][]; verified: number; latency_ms: number; telemetry: WireResponse["telemetry"] };
    return { candidates: r.candidates.map(([position, alt, delta]) => ({ position, alt, delta })), verified: r.verified, latencyMs: r.latency_ms, telemetry: telemetryFromWire(r.telemetry) };
  }
  close() { this.ws.close(); }
}
