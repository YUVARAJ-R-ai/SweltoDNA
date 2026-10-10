import { createStore } from "zustand/vanilla";
import type { Base, Region } from "./splice/sequence";
import type { SpliceClient, Telemetry, WindowScores } from "./splice/client";
import { deltaBetween, type Delta } from "./splice/scorer";

/** Bases shown on the 3D helix at once. */
export const WINDOW = 64;
/** Track span requested around the selection (covers the ribbon's visible range). */
export const TRACK_SPAN = 2048;

export interface Edit { position: number; from: Base; to: Base }

export interface ExplorerState {
  region: Region;
  engine: string;
  seq: Base[];
  selected: number;
  windowStart: number;
  edits: Edit[];
  delta: Delta | null;
  tracks: WindowScores | null;
  telemetry: Telemetry | null;
  latencyMs: number | null;
  scan: Float32Array;          // best |Δ| per position from the saturation scan, -1 = not scanned
  scanAlt: (Base | null)[];
  busy: boolean;
  error: string | null;
  init(): Promise<void>;
  select(i: number, opts?: { center?: boolean }): void;
  mutate(i: number, b: Base): Promise<void>;
  undo(): Promise<void>;
  reset(): Promise<void>;
  setScan(i: number, best: number, alt: Base): void;
  setScanMany(entries: [number, number, Base][]): void;
  setTelemetry(t: Telemetry | null): void;
  /** Δ for any edited position, from the current tracks (null if unedited or outside the tracks). */
  deltaAt(i: number): Delta | null;
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

export function createExplorerStore(region: Region, client: SpliceClient) {
  const n = region.seq.length;
  const trackWindow = (i: number) => { const s = clamp(i - TRACK_SPAN / 2, 0, Math.max(0, n - TRACK_SPAN)); return [s, Math.min(n, s + TRACK_SPAN)] as const; };
  const deltaFrom = (t: WindowScores, i: number): Delta => {
    // deltaBetween indexes absolute positions; lift the window into full-length views
    const lift = (a: Float32Array) => { const f = new Float32Array(n); f.set(a, t.start); return f; };
    return deltaBetween(lift(t.refDonor), lift(t.refAcceptor), lift(t.mutDonor), lift(t.mutAcceptor), i);
  };

  return createStore<ExplorerState>()((set, get) => {
    const apply = async (i: number, b: Base, record: Edit | null, pop = false) => {
      const s = get();
      const [ws, we] = trackWindow(i);
      set({ busy: true, error: null });
      try {
        const r = await client.mutate({ position: i, ref: region.seq[i], alt: b, windowStart: ws, windowEnd: we });
        const seq = s.seq.slice(); seq[i] = b;
        const edits = record ? [...get().edits, record] : pop ? get().edits.slice(0, -1) : get().edits;
        const last = edits[edits.length - 1];
        set({
          seq, edits, tracks: r.scores, telemetry: r.telemetry, latencyMs: r.latencyMs, engine: client.engine, busy: false,
          delta: last ? deltaFrom(r.scores, last.position) : null,
        });
      } catch (e) {
        const message = (e as Error).message;
        if (/cancel/i.test(message)) {
          // The server applied the edit but skipped scoring it because a newer edit superseded it.
          // Keep client state in step with the server; the newer edit's response brings fresh tracks.
          const seq = get().seq.slice(); seq[i] = b;
          const edits = record ? [...get().edits, record] : pop ? get().edits.slice(0, -1) : get().edits;
          set({ seq, edits, busy: false });
          return;
        }
        set({ busy: false, error: message });
      }
    };

    return {
      region, engine: client.engine, seq: region.seq.slice(),
      selected: region.feature, windowStart: region.feature - WINDOW / 2,
      edits: [], delta: null, tracks: null, telemetry: null, latencyMs: null,
      scan: new Float32Array(n).fill(-1), scanAlt: new Array(n).fill(null),
      busy: false, error: null,

      async init() {
        const [ws, we] = trackWindow(get().selected);
        set({ tracks: await client.score(ws, we), engine: client.engine });
      },
      select(i, opts) {
        const sel = clamp(Math.round(i), 0, n - 1), w0 = get().windowStart;
        const recentre = opts?.center || sel < w0 + 4 || sel >= w0 + WINDOW - 4;
        set({ selected: sel, windowStart: recentre ? clamp(sel - WINDOW / 2, 0, n - WINDOW) : w0 });
        const t = get().tracks;
        if (t && (sel < t.start + 128 || sel > t.end - 128)) {
          const [ws, we] = trackWindow(sel);
          client.score(ws, we).then((tracks) => set({ tracks })).catch((e) => set({ error: (e as Error).message }));
        }
      },
      async mutate(i, b) {
        const from = get().seq[i];
        if (from === b) return;
        get().select(i);
        await apply(i, b, { position: i, from, to: b });
      },
      async undo() {
        const last = get().edits[get().edits.length - 1];
        if (last) await apply(last.position, last.from, null, true);
      },
      async reset() {
        await client.reset();
        const [ws, we] = trackWindow(get().selected);
        set({ seq: region.seq.slice(), edits: [], delta: null, tracks: await client.score(ws, we) });
      },
      setScan(i, best, alt) {
        const scan = get().scan.slice(), scanAlt = get().scanAlt.slice();
        scan[i] = best; scanAlt[i] = alt;
        set({ scan, scanAlt });
      },
      setScanMany(entries) {
        const scan = get().scan.slice(), scanAlt = get().scanAlt.slice();
        for (const [i, best, alt] of entries) { scan[i] = best; scanAlt[i] = alt; }
        set({ scan, scanAlt });
      },
      setTelemetry(telemetry) { set({ telemetry }); },
      deltaAt(i) {
        const { seq, tracks } = get();
        if (seq[i] === region.seq[i] || !tracks || i < tracks.start || i >= tracks.end) return null;
        return deltaFrom(tracks, i);
      },
    };
  });
}

export type ExplorerStore = ReturnType<typeof createExplorerStore>;
