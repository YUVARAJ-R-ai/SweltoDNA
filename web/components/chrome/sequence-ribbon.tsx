"use client";
import { useEffect, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useExplorerCtx, useExplorer, useLevel } from "@/lib/explorer-context";
import { impactTier } from "@/lib/splice/scorer";
import { WINDOW } from "@/lib/store";
import { heatCss } from "@/scene/engine";

const BW = 20;            // px per base
const TRACK_H = 34;       // two 16 px rows (#8 reference / mutant) + gap

interface Tip { x: number; y: number; text: string; color: string }

/**
 * #7 virtualized ribbon (only visible chips exist in the DOM) with the #8 dual track underneath.
 * Both live in one scroll container, so they stay aligned by construction.
 */
export function SequenceRibbon({ onPick }: { onPick: (i: number, x: number, y: number) => void }) {
  const { store } = useExplorerCtx();
  const level = useLevel();
  const { seq, region, selected, windowStart, scan, tracks, edits } = useExplorer((s) => s);
  const scrollRef = useRef<HTMLDivElement>(null), trackRef = useRef<HTMLCanvasElement>(null), overviewRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const v = useVirtualizer({ horizontal: true, count: seq.length, getScrollElement: () => scrollRef.current, estimateSize: () => BW, overscan: 10 });
  const items = v.getVirtualItems();
  const first = items[0]?.index ?? 0, last = items[items.length - 1]?.index ?? 0;

  useEffect(() => {                                         // keep the selection in view
    const el = scrollRef.current; if (!el) return;
    const x = selected * BW;
    if (x < el.scrollLeft + 48 || x > el.scrollLeft + el.clientWidth - 48) {
      el.scrollTo({ left: x - el.clientWidth / 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  }, [selected]);

  useEffect(() => {                                         // #8 dual track for the visible range
    const cv = trackRef.current; if (!cv) return;
    const n = last - first + 1, dpr = devicePixelRatio || 1, W = n * BW;
    cv.width = Math.floor(W * dpr); cv.height = Math.floor(TRACK_H * dpr); cv.style.width = `${W}px`; cv.style.left = `${first * BW}px`;
    const g = cv.getContext("2d"); if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, TRACK_H);
    for (let i = first; i <= last; i++) {
      const x = (i - first) * BW + 2, w = BW - 4;
      g.fillStyle = "rgba(120,120,128,.16)"; g.fillRect(x, 15, w, 1); g.fillRect(x, 33, w, 1);
      if (!tracks || i < tracks.start || i >= tracks.end) continue;
      const k = i - tracks.start;
      const ref = Math.max(tracks.refDonor[k], tracks.refAcceptor[k]), mut = Math.max(tracks.mutDonor[k], tracks.mutAcceptor[k]);
      if (ref > 0.01) { g.fillStyle = "#0A84FF"; g.fillRect(x, 15 - ref * 14, w, ref * 14); }          // reference: blue
      const diff = mut - ref;
      if (diff > 0.05) { g.fillStyle = "#FF9F0A"; g.fillRect(x, 33 - mut * 14, w, mut * 14); }          // gain: amber spike
      else if (diff < -0.05) { g.strokeStyle = "#FF453A"; g.lineWidth = 1.5; g.strokeRect(x + 0.75, 33 - ref * 14 + 0.75, w - 1.5, ref * 14 - 1.5); if (mut > 0.01) { g.fillStyle = "#FF453A"; g.fillRect(x, 33 - mut * 14, w, mut * 14); } }  // loss: red outline of what was there
      else if (mut > 0.01) { g.fillStyle = "rgba(235,235,245,.5)"; g.fillRect(x, 33 - mut * 14, w, mut * 14); }
    }
  }, [first, last, tracks]);

  const hover = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), i = first + Math.floor((e.clientX - r.left) / BW);
    if (!tracks || i < tracks.start || i >= tracks.end) { setTip(null); return; }
    const k = i - tracks.start, dd = tracks.mutDonor[k] - tracks.refDonor[k], da = tracks.mutAcceptor[k] - tracks.refAcceptor[k];
    const useD = Math.abs(dd) >= Math.abs(da), delta = useD ? dd : da;
    const cls = Math.abs(delta) < 0.01 ? "No change" : `${useD ? "Donor" : "Acceptor"} ${delta > 0 ? "Gain" : "Loss"}`;
    const tier = impactTier(Math.abs(delta));
    setTip({ x: e.clientX, y: r.top - 8, color: delta > 0.01 ? "var(--orange)" : delta < -0.01 ? "var(--red)" : "var(--label2)",
      text: `${region.chrom}:${(region.coord0 + i).toLocaleString("en-US")} · ${cls} · Δ ${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(2)} · ${tier}` });
  };

  const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = overviewRef.current!.getBoundingClientRect(), i = Math.max(0, Math.min(seq.length - 1, Math.floor(((e.clientX - r.left) / r.width) * seq.length)));
    store.getState().select(i, { center: true });
    if (scrollRef.current) scrollRef.current.scrollLeft = i * BW - scrollRef.current.clientWidth / 2;
  };
  const edited = Array.from(new Set(edits.map((e) => e.position))).filter((i) => seq[i] !== region.seq[i]);

  return (
    <section aria-label="Sequence" data-off={level < 2} data-scroll-own
      className="sv-mat sv-regular sv-panel fixed bottom-4 left-4 right-4 z-[5] rounded-[22px] px-3.5 pb-2.5 pt-3 data-[off=true]:translate-y-[18px] data-[off=true]:scale-[.99]">
      <div ref={overviewRef} className="relative h-2.5 cursor-pointer touch-none rounded-[5px] bg-[var(--fill2)]" role="slider" tabIndex={0}
        aria-label="Position in region" aria-valuemin={0} aria-valuemax={seq.length - 1} aria-valuenow={selected}
        aria-valuetext={`${region.chrom}:${(region.coord0 + selected).toLocaleString("en-US")}`}
        onKeyDown={(e) => { const step = e.shiftKey ? 1000 : 100; if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); store.getState().select(selected + (e.key === "ArrowRight" ? step : -step), { center: true }); } }}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); scrub(e); }} onPointerMove={(e) => { if (e.buttons) scrub(e); }}>
        {region.exons.map(([s, e]) => <div key={s} className="absolute bottom-0.5 top-0.5 rounded-sm bg-[var(--label3)]" style={{ left: `${(s / seq.length) * 100}%`, width: `${((e - s) / seq.length) * 100}%` }} />)}
        <div className="absolute -bottom-[3px] -top-[3px] rounded-[5px] border-[1.5px] border-[var(--blue)] bg-[rgba(10,132,255,.18)]" style={{ left: `${(windowStart / seq.length) * 100}%`, width: `${Math.max(0.6, (WINDOW / seq.length) * 100)}%` }} />
        {edited.map((i) => <div key={i} className="absolute -top-[3px] -ml-[3px] h-1.5 w-1.5 rounded-full bg-[var(--orange)]" style={{ left: `${(i / seq.length) * 100}%` }} />)}
      </div>

      <div ref={scrollRef} data-testid="ribbon" tabIndex={0} aria-label="Sequence ribbon" role="region" className="relative mt-2.5 overflow-x-auto overflow-y-hidden [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        style={{ height: 50 + TRACK_H + 6, maskImage: "linear-gradient(90deg,transparent,#000 28px,#000 calc(100% - 28px),transparent)", WebkitMaskImage: "linear-gradient(90deg,transparent,#000 28px,#000 calc(100% - 28px),transparent)" }}>
        <div className="relative h-full" style={{ width: v.getTotalSize() }}>
          <div className="pointer-events-none absolute top-3 h-[34px] rounded-[9px] bg-[rgba(10,132,255,.12)] shadow-[inset_0_0_0_1px_rgba(10,132,255,.5)]" style={{ left: windowStart * BW - 2, width: WINDOW * BW }} />
          {items.map((it) => {
            const i = it.index, b = seq[i];
            return (
              <div key={it.key}>
                {i % 10 === 0 && <div className="sv-mono absolute top-0 -translate-x-1/2 text-[10px] text-[var(--label3)]" style={{ left: it.start + 9 }}>{i % 50 === 0 ? (region.coord0 + i).toLocaleString("en-US") : "·"}</div>}
                {scan[i] >= 0.05 && <div className="absolute top-[11px] h-[3px] w-[18px] rounded-sm" style={{ left: it.start, background: heatCss(scan[i]), opacity: 0.35 + scan[i] * 0.65 }} />}
                <div className="sv-chip" data-testid="chip" data-i={i} data-sel={i === selected} data-mut={b !== region.seq[i]} style={{ left: it.start, background: `var(--${b})` }}
                  onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); store.getState().select(i, { center: true }); onPick(i, r.left + r.width / 2, r.top - 84); }}>{b}</div>
              </div>
            );
          })}
          <canvas ref={trackRef} data-testid="dual-track" role="img" aria-label="Reference (blue) and mutant (amber gain, red loss) splice tracks" className="absolute" style={{ top: 50, height: TRACK_H }}
            onMouseMove={hover} onMouseLeave={() => setTip(null)} />
        </div>
      </div>
      <div className="mt-1.5 flex justify-between gap-3 text-[11px] text-[var(--label3)]">
        <span>Reference <span className="text-[var(--blue)]">■</span> · mutant gain <span className="text-[var(--orange)]">■</span> · loss <span className="text-[var(--red)]">□</span> · synthetic sequence, illustrative coordinates</span>
        <span className="sv-mono">{(region.coord0 + first).toLocaleString("en-US")}–{(region.coord0 + last).toLocaleString("en-US")} · {items.length} of {seq.length.toLocaleString()} in DOM</span>
      </div>
      {tip && <div role="tooltip" data-testid="track-tooltip" className="sv-mat sv-thick pointer-events-none fixed z-[9] -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-[10px] px-2.5 py-1.5 text-xs font-medium" style={{ left: tip.x, top: tip.y, color: tip.color }}>{tip.text}</div>}
    </section>
  );
}
