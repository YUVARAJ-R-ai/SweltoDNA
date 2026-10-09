import { describe, expect, it } from "vitest";
import { syntheticRegion } from "@/lib/splice/sequence";
import { SpliceModel, consequence, impactTier, describeDelta } from "@/lib/splice/scorer";

const region = syntheticRegion();
const F = region.feature; // exon 10 donor G

describe("reference scores", () => {
  const m = new SpliceModel(region);
  it("scores the featured donor and acceptor as strong sites", () => {
    expect(m.refDonor[F]).toBeCloseTo(1.0, 2);
    expect(m.refAcceptor[4868]).toBeCloseTo(0.91, 2);
  });
  it("suppresses the latent cryptic donor while the canonical donor is intact", () => {
    expect(m.refDonor[F + 17]).toBeLessThan(0.05);
  });
});

describe("mutation deltas", () => {
  it("donor +2 T>A abolishes the donor and activates the cryptic donor 16 bp downstream", () => {
    const m = new SpliceModel(region);
    const d = m.mutate(F + 1, "A");
    expect(d.donorLoss).toBeCloseTo(1.0, 2);
    expect(d.donorLossAt).toBe(F);
    expect(d.donorGain).toBeCloseTo(0.62, 2);
    expect(d.donorGainAt - (F + 1)).toBe(16);
  });
  it("acceptor A>C abolishes the acceptor", () => {
    const m = new SpliceModel(region);
    const d = m.mutate(4868, "C");
    expect(d.acceptorLoss).toBeCloseTo(0.91, 2);
  });
  it("a deep intronic change is minimal", () => {
    const m = new SpliceModel(region);
    expect(m.mutate(5400, "G").max).toBeLessThan(0.2);
  });
  it("undo restores reference scores", () => {
    const m = new SpliceModel(region);
    m.mutate(F + 1, "A");
    m.set(F + 1, region.seq[F + 1]);
    expect(m.delta(F + 1).max).toBe(0);
  });
});

describe("interpretation", () => {
  it("predicts exon 10 extension by 17 nt with a frameshift", () => {
    const m = new SpliceModel(region);
    const c = consequence(m.mutate(F + 1, "A"), region);
    expect(c).toMatchObject({ kind: "shift", side: "donor", nt: 17, exonNumber: 10, frameshift: true });
  });
  it("predicts exon skipping when the acceptor is lost without a replacement", () => {
    const m = new SpliceModel(region);
    const c = consequence(m.mutate(4868, "C"), region);
    expect(c).toMatchObject({ kind: "skip", exonNumber: 10, frameshift: true });
  });
  it("uses issue #8 tiers", () => {
    expect(impactTier(0.81)).toBe("High");
    expect(impactTier(0.8)).toBe("Moderate");
    expect(impactTier(0.5)).toBe("Low");
    expect(impactTier(0.2)).toBe("Minimal");
  });
  it("describes the change in plain language", () => {
    const m = new SpliceModel(region);
    expect(describeDelta(m.mutate(F + 1, "A"), F + 1)).toBe(
      "Abolishes the donor 1 bp upstream and activates a cryptic donor 16 bp downstream.",
    );
  });
});

describe("saturation scan", () => {
  it("ranks the canonical donor positions highest", () => {
    const m = new SpliceModel(region);
    const r = m.scanPosition(F);
    expect(r.best).toBeCloseTo(1.0, 2);
    expect(m.scanPosition(F + 4).best).toBeCloseTo(0.62, 1);
  });
});
