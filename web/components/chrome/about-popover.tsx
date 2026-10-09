"use client";
import { useExplorer, useUI } from "@/lib/explorer-context";

export function AboutPopover() {
  const open = useUI((s) => s.aboutOpen);
  const engine = useExplorer((s) => s.engine);
  return (
    <div role="dialog" aria-label="About this demo" data-off={!open}
      className="sv-mat sv-thick sv-panel fixed right-4 top-[68px] z-[7] w-[340px] origin-[calc(100%-22px)_-8px] rounded-[18px] px-[18px] py-4 text-[13px] leading-[1.45] text-[var(--label2)] data-[off=true]:scale-[.94]">
      <h3 className="mb-2 text-[15px] font-semibold tracking-[-0.01em] text-[var(--label)]">About this demo</h3>
      <p><b className="font-semibold text-[var(--label)]">Sequence.</b> A synthetic 10 kb region with planted exons. The MAPT label and genomic coordinates are illustrative.</p>
      <p className="mt-2"><b className="font-semibold text-[var(--label)]">Splice scores.</b> Computed by <b className="font-semibold text-[var(--label)]">{engine}</b>. The built-in heuristic is a motif scorer with local site competition, not a trained model.</p>
      <p className="mt-2"><b className="font-semibold text-[var(--label)]">Scan and consequences.</b> The scan scores all three alternatives at every position on the reference. Exon extension, shortening, or skipping is inferred from where the strongest site moves. Both are heuristic.</p>
      <p className="mt-2"><b className="font-semibold text-[var(--label)]">Performance.</b> Speculative-verification figures are labelled simulated until the verifier reports measured values (issue #4).</p>
    </div>
  );
}
