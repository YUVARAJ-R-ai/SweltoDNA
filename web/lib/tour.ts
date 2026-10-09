import type { ExplorerCtx } from "./explorer-context";
import { consequence } from "./splice/scorer";

/** Guided tour. Any user input aborts it (agency); every step checks the signal. */
export async function runTour(ctx: ExplorerCtx, signal: AbortSignal, actions: { mutate: (i: number, b: "A" | "C" | "G" | "T") => Promise<void>; scan: () => void }) {
  const { store, engine, island } = ctx;
  const F = store.getState().region.feature;
  const sleep = (ms: number) => new Promise<void>((res, rej) => {
    const t = setTimeout(res, ms);
    signal.addEventListener("abort", () => { clearTimeout(t); rej(new DOMException("aborted", "AbortError")); }, { once: true });
  });
  const step = async (ms: number, text: string, fn?: () => unknown) => {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    await fn?.();
    island.show(text, { hold: ms + 500, wrap: true });
    await sleep(ms);
  };
  await store.getState().reset();
  engine.current?.resetOrbit();
  await step(4800, "Somewhere in this brain, one letter of DNA decides how a gene is read.", () => { engine.current!.zoomTarget = 0; });
  await step(4800, "Chromosome 17. Band q21.31 holds MAPT, the gene for the tau protein.", () => { engine.current!.zoomTarget = 1; });
  await step(5200, "Exon 10. The GT at its edge is a donor site: it tells the cell where the exon ends.", () => { engine.current!.zoomTarget = 2; store.getState().select(F, { center: true }); });
  await step(3800, "Change one letter, T → A, and the donor is gone.", () => actions.mutate(F + 1, "A"));
  const d = store.getState().deltaAt(F + 1), c = d ? consequence(d, store.getState().region) : null;
  const msg = c && c.kind === "shift"
    ? `A weaker site ${d!.donorGainAt - (F + 1)} bp downstream takes over. Exon 10 grows by ${c.nt} nt, and the reading frame ${c.frameshift ? "shifts" : "holds"}.`
    : "The splicing pattern changes.";
  await step(5800, msg, () => { engine.current!.zoomTarget = 3; });
  await step(4600, "Now every possible single-letter change in view, scored live in your browser.", () => { engine.current!.zoomTarget = 2; actions.scan(); });
  await step(3400, "Your turn. Click any rung to rewrite it.");
}
