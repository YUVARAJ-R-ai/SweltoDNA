/**
 * Heuristic splice scorer: consensus-motif strength with local same-type site competition.
 * Not a trained model; the UI labels its output "Heuristic".
 */
import type { Base, Region } from "./sequence";
import { BASES } from "./sequence";

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const smoothstep = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const COMPETITION_BP = 40;
export const DELTA_WINDOW_BP = 50;

type Seq = ArrayLike<string>;

function donorRaw(s: Seq, i: number): number {
  if (s[i] !== "G") return 0;
  const w = s[i + 1] === "T" ? 1 : s[i + 1] === "C" ? 0.5 : 0;
  if (!w) return 0;
  return w * (0.15 + (s[i - 1] === "G" ? 0.12 : 0) + (s[i - 2] === "A" ? 0.08 : 0) + (s[i - 3] === "A" || s[i - 3] === "C" ? 0.05 : 0)
    + (s[i + 2] === "A" || s[i + 2] === "G" ? 0.12 : 0) + (s[i + 3] === "A" ? 0.12 : 0) + (s[i + 4] === "G" ? 0.2 : 0) + (s[i + 5] === "T" ? 0.06 : 0));
}

function acceptorRaw(s: Seq, i: number): number {
  if (s[i] !== "A" || s[i + 1] !== "G") return 0;
  let py = 0;
  for (let k = i - 14; k <= i - 3; k++) if (s[k] === "C" || s[k] === "T") py++;
  return 0.15 + 0.55 * Math.pow(py / 12, 1.5) + (s[i - 1] === "C" || s[i - 1] === "T" ? 0.1 : 0) + (s[i + 2] === "G" ? 0.1 : 0);
}

/** Writes donor/acceptor probabilities for positions [lo, hi) into D and A. */
export function scoreRange(s: Seq, lo: number, hi: number, D: Float32Array, A: Float32Array): void {
  const n = s.length;
  lo = clamp(lo, 0, n); hi = clamp(hi, 0, n);
  const L0 = Math.max(0, lo - COMPETITION_BP), H0 = Math.min(n, hi + COMPETITION_BP);
  const rd = new Float32Array(H0 - L0), ra = new Float32Array(H0 - L0);
  for (let i = L0; i < H0; i++) { rd[i - L0] = donorRaw(s, i); ra[i - L0] = acceptorRaw(s, i); }
  const one = (raw: Float32Array, i: number) => {
    const r = raw[i - L0];
    if (r < 0.45) return 0;
    let mx = 0;
    for (let k = Math.max(L0, i - COMPETITION_BP); k < Math.min(H0, i + COMPETITION_BP + 1); k++) {
      if (Math.abs(k - i) > 2 && raw[k - L0] > mx) mx = raw[k - L0];
    }
    return smoothstep(0.45, 0.88, r) * (1 - 0.95 * clamp((mx - r) * 6, 0, 1));
  };
  for (let i = lo; i < hi; i++) { D[i] = one(rd, i); A[i] = one(ra, i); }
}

export interface Delta {
  donorLoss: number; donorGain: number; acceptorLoss: number; acceptorGain: number;
  donorLossAt: number; donorGainAt: number; acceptorLossAt: number; acceptorGainAt: number;
  max: number;
}

export function deltaBetween(refD: Float32Array, refA: Float32Array, curD: Float32Array, curA: Float32Array, v: number): Delta {
  const d: Delta = { donorLoss: 0, donorGain: 0, acceptorLoss: 0, acceptorGain: 0, donorLossAt: v, donorGainAt: v, acceptorLossAt: v, acceptorGainAt: v, max: 0 };
  for (let i = Math.max(0, v - DELTA_WINDOW_BP); i <= Math.min(refD.length - 1, v + DELTA_WINDOW_BP); i++) {
    const dd = curD[i] - refD[i], da = curA[i] - refA[i];
    if (-dd > d.donorLoss) { d.donorLoss = -dd; d.donorLossAt = i; }
    if (dd > d.donorGain) { d.donorGain = dd; d.donorGainAt = i; }
    if (-da > d.acceptorLoss) { d.acceptorLoss = -da; d.acceptorLossAt = i; }
    if (da > d.acceptorGain) { d.acceptorGain = da; d.acceptorGainAt = i; }
  }
  d.max = Math.max(d.donorLoss, d.donorGain, d.acceptorLoss, d.acceptorGain);
  return d;
}

/** Reference scores plus a mutable current sequence with incrementally updated scores. */
export class SpliceModel {
  readonly region: Region;
  readonly seq: Base[];
  readonly refDonor: Float32Array; readonly refAcceptor: Float32Array;
  readonly curDonor: Float32Array; readonly curAcceptor: Float32Array;
  private readonly scratchD: Float32Array; private readonly scratchA: Float32Array;
  private readonly scratchSeq: Base[];

  constructor(region: Region) {
    this.region = region;
    const n = region.seq.length;
    this.seq = region.seq.slice();
    this.scratchSeq = region.seq.slice();
    this.refDonor = new Float32Array(n); this.refAcceptor = new Float32Array(n);
    scoreRange(region.seq, 0, n, this.refDonor, this.refAcceptor);
    this.curDonor = this.refDonor.slice(); this.curAcceptor = this.refAcceptor.slice();
    this.scratchD = new Float32Array(n); this.scratchA = new Float32Array(n);
  }

  get length() { return this.seq.length; }

  /** Sets one base and rescores the neighbourhood that can be affected. */
  set(i: number, b: Base): void {
    this.seq[i] = b;
    const reach = DELTA_WINDOW_BP + 2 * COMPETITION_BP + 20;
    scoreRange(this.seq, i - reach, i + reach, this.curDonor, this.curAcceptor);
  }

  mutate(i: number, b: Base): Delta { this.set(i, b); return this.delta(i); }

  delta(i: number): Delta { return deltaBetween(this.refDonor, this.refAcceptor, this.curDonor, this.curAcceptor, i); }

  reset(): void {
    for (let i = 0; i < this.seq.length; i++) this.seq[i] = this.region.seq[i];
    this.curDonor.set(this.refDonor); this.curAcceptor.set(this.refAcceptor);
  }

  /** In-silico saturation mutagenesis at one position, on the reference: the strongest |Δ| over all alternatives. */
  scanPosition(i: number): { best: number; alt: Base } {
    let best = 0, alt: Base = BASES.find((b) => b !== this.region.seq[i])!;
    const s = this.scratchSeq;
    for (const b of BASES) {
      if (b === this.region.seq[i]) continue;
      s[i] = b;
      scoreRange(s, i - DELTA_WINDOW_BP, i + DELTA_WINDOW_BP + 1, this.scratchD, this.scratchA);
      let m = 0;
      for (let j = Math.max(0, i - DELTA_WINDOW_BP); j < Math.min(s.length, i + DELTA_WINDOW_BP + 1); j++) {
        m = Math.max(m, Math.abs(this.scratchD[j] - this.refDonor[j]), Math.abs(this.scratchA[j] - this.refAcceptor[j]));
      }
      if (m >= best) { best = m; alt = b; }
    }
    s[i] = this.region.seq[i];
    return { best, alt };
  }
}

export type Tier = "High" | "Moderate" | "Low" | "Minimal";
/** Issue #8 tiers: High Δ>0.8, Moderate 0.5<Δ≤0.8, Low 0.2<Δ≤0.5. */
export function impactTier(m: number): Tier { return m > 0.8 ? "High" : m > 0.5 ? "Moderate" : m > 0.2 ? "Low" : "Minimal"; }

const offset = (o: number) => (o === 0 ? "at the variant" : `${Math.abs(o)} bp ${o > 0 ? "downstream" : "upstream"}`);

export function describeDelta(d: Delta, i: number): string {
  if (d.max < 0.2) return "No meaningful change to splice sites within 50 bp.";
  const p: string[] = [];
  if (d.donorLoss >= 0.2) p.push(`${d.donorLoss >= 0.5 ? "abolishes" : "weakens"} the donor ${d.donorLossAt === i ? "here" : offset(d.donorLossAt - i)}`);
  if (d.acceptorLoss >= 0.2) p.push(`${d.acceptorLoss >= 0.5 ? "abolishes" : "weakens"} the acceptor ${offset(d.acceptorLossAt - i)}`);
  if (d.donorGain >= 0.2) p.push(`activates a cryptic donor ${offset(d.donorGainAt - i)}`);
  if (d.acceptorGain >= 0.2) p.push(`activates a cryptic acceptor ${offset(d.acceptorGainAt - i)}`);
  const s = p.join(p.length > 2 ? ", " : " and ");
  return s.charAt(0).toUpperCase() + s.slice(1) + ".";
}

export type Consequence =
  | { kind: "shift"; side: "donor" | "acceptor"; nt: number; exonIndex: number; exonNumber: number; frameshift: boolean }
  | { kind: "skip"; exonIndex: number; exonNumber: number; exonLength: number; frameshift: boolean }
  | { kind: "gain"; site: "donor" | "acceptor" };

/** Infers the transcript outcome from where the strongest site moves. Heuristic, and labelled as such. */
export function consequence(d: Delta, region: Region): Consequence | null {
  const ex = region.exons, num = (k: number) => k + region.firstExonNumber;
  const inner = (k: number) => k > 0 && k < ex.length - 1;
  if (d.donorLoss >= 0.5) {
    const k = ex.findIndex(([, e]) => e === d.donorLossAt);
    if (inner(k)) {
      if (d.donorGain >= 0.3) { const nt = d.donorGainAt - d.donorLossAt; return { kind: "shift", side: "donor", nt, exonIndex: k, exonNumber: num(k), frameshift: nt % 3 !== 0 }; }
      const len = ex[k][1] - ex[k][0];
      return { kind: "skip", exonIndex: k, exonNumber: num(k), exonLength: len, frameshift: len % 3 !== 0 };
    }
  }
  if (d.acceptorLoss >= 0.5) {
    const k = ex.findIndex(([s]) => s - 2 === d.acceptorLossAt);
    if (inner(k)) {
      if (d.acceptorGain >= 0.3) { const nt = d.acceptorLossAt - d.acceptorGainAt; return { kind: "shift", side: "acceptor", nt, exonIndex: k, exonNumber: num(k), frameshift: nt % 3 !== 0 }; }
      const len = ex[k][1] - ex[k][0];
      return { kind: "skip", exonIndex: k, exonNumber: num(k), exonLength: len, frameshift: len % 3 !== 0 };
    }
  }
  if (d.donorGain >= 0.5) return { kind: "gain", site: "donor" };
  if (d.acceptorGain >= 0.5) return { kind: "gain", site: "acceptor" };
  return null;
}

/** "exon 10 donor +2", "acceptor −1", "exon 10, base 5 of 130", or "intron". */
export function siteContext(i: number, region: Region): string {
  for (let k = 0; k < region.exons.length; k++) {
    const [s, e] = region.exons[k], n = k + region.firstExonNumber;
    if (i >= e && i <= e + 5) return `exon ${n} donor +${i - e + 1}`;
    if (i >= s - 3 && i < s) return `acceptor −${s - i}`;
    if (i >= s && i < e) return `exon ${n}, base ${i - s + 1} of ${e - s}`;
  }
  return "intron";
}
