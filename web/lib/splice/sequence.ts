export type Base = "A" | "C" | "G" | "T";
export const BASES: readonly Base[] = ["A", "C", "G", "T"];
export const COMPLEMENT: Record<Base, Base> = { A: "T", T: "A", C: "G", G: "C" };

export interface Region {
  /** Reference sequence (never mutated). */
  seq: readonly Base[];
  /** [start, end) of each exon; the donor G sits at `end`, the acceptor A at `start - 2`. */
  exons: readonly (readonly [number, number])[];
  /** Number shown for exons[0]; with 5, the featured exons[5] is "exon 10". */
  firstExonNumber: number;
  /** Donor G of the featured exon. */
  feature: number;
  coord0: number;
  chrom: string;
  gene: string;
  synthetic: boolean;
  description?: string;
}

/** Region as sent by the #6 server's hello message. */
export interface WireRegion {
  length: number; chrom: string; gene: string; coord0: number; feature: number;
  exons: [number, number][]; first_exon_number: number; synthetic: boolean; description: string; sequence: string;
}
export function regionFromWire(w: WireRegion): Region {
  return { seq: w.sequence.split("") as Base[], exons: w.exons.map(([s, e]) => [s, e] as const), firstExonNumber: w.first_exon_number,
    feature: w.feature, coord0: w.coord0, chrom: w.chrom, gene: w.gene, synthetic: w.synthetic, description: w.description };
}

function parkMiller(seed: number) {
  return () => (seed = (seed * 16807) % 2147483647) / 2147483647;
}

/**
 * Deterministic 10 kb demo region with planted exons, consensus splice sites, and a latent cryptic
 * donor 17 bp downstream of the featured exon's donor. Labelled synthetic everywhere it is shown.
 */
export function syntheticRegion(): Region {
  const LEN = 10000;
  const r = parkMiller(1234);
  const seq = Array.from({ length: LEN }, () => BASES[Math.floor(r() * 4)]);
  const plant = (at: number, s: string) => { for (let k = 0; k < s.length; k++) seq[at + k] = s[k] as Base; };
  const exons: [number, number][] = [[620, 760], [1500, 1640], [2400, 2510], [3300, 3460], [4100, 4240], [4870, 5000],
    [5800, 5950], [6700, 6830], [7600, 7790], [8500, 8620], [9300, 9420]];
  for (const [s, e] of exons) { plant(s - 18, "TTTCTTTTCCCTTTTCAG"); plant(e - 3, "CAGGTAAGT"); }
  const feature = 5000;
  plant(feature + 17 - 3, "TTGGTGAGC");
  return { seq, exons, firstExonNumber: 5, feature, coord0: 45960000, chrom: "chr17", gene: "MAPT", synthetic: true };
}
