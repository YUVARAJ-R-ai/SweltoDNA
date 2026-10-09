import { beforeEach, describe, expect, it } from "vitest";
import { createExplorerStore, WINDOW } from "@/lib/store";
import { syntheticRegion } from "@/lib/splice/sequence";
import { LocalHeuristicClient } from "@/lib/splice/client";

const region = syntheticRegion();
const F = region.feature;
let store: ReturnType<typeof createExplorerStore>;

beforeEach(async () => {
  store = createExplorerStore(region, new LocalHeuristicClient(region));
  await store.getState().init();
});

describe("explorer store", () => {
  it("starts on the featured donor with the helix window centred on it", () => {
    const s = store.getState();
    expect(s.selected).toBe(F);
    expect(s.windowStart).toBe(F - WINDOW / 2);
  });
  it("mutation records an edit and a high-impact delta", async () => {
    await store.getState().mutate(F + 1, "A");
    const s = store.getState();
    expect(s.seq[F + 1]).toBe("A");
    expect(s.edits).toEqual([{ position: F + 1, from: "T", to: "A" }]);
    expect(s.delta!.donorLoss).toBeCloseTo(1, 2);
  });
  it("undo restores the base and clears the delta", async () => {
    await store.getState().mutate(F + 1, "A");
    await store.getState().undo();
    const s = store.getState();
    expect(s.seq[F + 1]).toBe("T");
    expect(s.edits).toEqual([]);
    expect(s.delta).toBeNull();
  });
  it("re-centres the helix window when the selection leaves it", () => {
    store.getState().select(F + 200);
    expect(store.getState().windowStart).toBe(F + 200 - WINDOW / 2);
  });
  it("keeps selection inside the region", () => {
    store.getState().select(-5);
    expect(store.getState().selected).toBe(0);
    store.getState().select(1e9);
    expect(store.getState().selected).toBe(region.seq.length - 1);
  });
});

describe("deltaAt and scan", () => {
  it("returns a delta only for edited positions", async () => {
    expect(store.getState().deltaAt(F + 1)).toBeNull();
    await store.getState().mutate(F + 1, "A");
    expect(store.getState().deltaAt(F + 1)!.donorGain).toBeCloseTo(0.62, 2);
  });
  it("setScan publishes new arrays so subscribers re-render", () => {
    const before = store.getState().scan;
    store.getState().setScan(F, 1, "A");
    expect(store.getState().scan).not.toBe(before);
    expect(store.getState().scan[F]).toBe(1);
  });
});
