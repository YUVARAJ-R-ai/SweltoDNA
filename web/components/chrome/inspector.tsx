"use client";
import { useMemo } from "react";
import { useExplorerCtx, useExplorer, useLevel, useUI } from "@/lib/explorer-context";
import { BASES, type Base } from "@/lib/splice/sequence";
import { consequence, describeDelta, impactTier, siteContext } from "@/lib/splice/scorer";
import { WINDOW } from "@/lib/store";
import { heatCss } from "@/scene/engine";
import { TranscriptDiagram } from "./transcript-diagram";

const TIER_COLOR = { High: "var(--red)", Moderate: "var(--orange)", Low: "var(--yellow)", Minimal: "var(--label2)" } as const;

export function Inspector({ onMutate, onScan }: { onMutate: (i: number, b: Base) => void; onScan: () => void }) {
  const { store } = useExplorerCtx();
  const level = useLevel();
  const { region, seq, selected, edits, tracks, windowStart, scan, scanAlt, engine } = useExplorer((s) => s);
  const scanning = useUI((s) => s.scanning);
  const snapOpen = useUI((s) => !!s.snapshot);
  const base = seq[selected], ref = region.seq[selected], changed = base !== ref;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const d = useMemo(() => store.getState().deltaAt(selected), [selected, tracks, seq]);
  const cons = d ? consequence(d, region) : null;
  const tier = d ? impactTier(d.max) : null;
  const top = useMemo(() => {
    const idx = Array.from({ length: WINDOW }, (_, k) => windowStart + k).filter((i) => scan[i] >= 0);
    return idx.sort((a, b) => scan[b] - scan[a]).slice(0, 5);
  }, [scan, windowStart]);
  const scanned = top.length > 0;
  const fmt = (i: number) => (region.coord0 + i).toLocaleString("en-US");
  const rows: [string, number, string][] = [["Donor loss", d?.donorLoss ?? 0, "var(--red)"], ["Donor gain", d?.donorGain ?? 0, "var(--orange)"], ["Acceptor loss", d?.acceptorLoss ?? 0, "var(--red)"], ["Acceptor gain", d?.acceptorGain ?? 0, "var(--orange)"]];

  return (
    <aside aria-label="Variant" data-off={level < 2 || snapOpen}
      className="sv-mat sv-thick sv-panel fixed right-4 top-[72px] z-[5] max-h-[calc(100vh-72px-200px)] w-[328px] overflow-y-auto rounded-[22px] px-[18px] pb-4 pt-[18px] data-[off=true]:translate-x-[18px] data-[off=true]:scale-[.98] max-[900px]:left-4 max-[900px]:w-auto">
      <div className="sv-eyebrow">{changed ? "Variant" : "Selected base"}</div>
      <div className="sv-mono mt-1 text-[30px] font-semibold leading-[1.08] tracking-[-0.02em]" data-testid="variant-title">{changed ? `${ref} → ${base}` : base}</div>
      <div className="sv-num text-[13px] text-[var(--label2)]">{region.chrom}:{fmt(selected)} · {siteContext(selected, region)}</div>

      <div className="sv-seg mt-3.5" role="group" aria-label="Base at this position">
        {BASES.map((b) => (
          <button key={b} aria-pressed={b === base} onClick={() => onMutate(selected, b)} className="sv-mono text-[15px] font-semibold">
            <i className="inline-block h-2 w-2 rounded-full" style={{ background: `var(--${b})` }} />{b}
          </button>
        ))}
      </div>

      <p className="mb-1 mt-3.5 text-[15px] leading-[1.4] tracking-[-0.005em]" data-testid="sentence">
        {d && tier ? <><span className="font-semibold" style={{ color: TIER_COLOR[tier] }}>{tier} impact.</span> {describeDelta(d, selected)}</> : "Choose another base to see how splicing changes."}
      </p>
      {cons && <TranscriptDiagram c={cons} />}

      <div className="mt-4">
        {rows.map(([name, v, color]) => (
          <div key={name} className="grid h-[30px] grid-cols-[1fr_84px_42px] items-center gap-2.5 text-[13px]">
            <span>{name}</span>
            <div className="h-1 overflow-hidden rounded-sm bg-[var(--fill2)]"><i className="block h-full rounded-sm transition-[width] duration-500" style={{ width: `${(v * 100).toFixed(0)}%`, background: color }} /></div>
            <span className="sv-mono text-right">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>

      <div className="sv-hr" />
      <div className="flex items-center justify-between gap-2.5">
        <span className="sv-eyebrow">Every variant in view</span>
        <button className="sv-pill sv-primary" disabled={scanning} onClick={onScan}>{scanning ? "Scanning…" : scanned ? "Rescan" : `Scan ${WINDOW * 3} variants`}</button>
      </div>
      {scanned && !scanning && (
        <div className="mt-1.5" data-testid="scan-results">
          {top.map((i) => (
            <button key={i} onClick={() => { store.getState().select(i); if (scanAlt[i]) onMutate(i, scanAlt[i]!); }}
              className="grid h-[30px] w-full grid-cols-[1fr_auto_44px] items-center gap-2.5 rounded-[9px] px-2.5 text-left text-xs hover:bg-[var(--fill2)]">
              <span>{siteContext(i, region)}</span>
              <span className="sv-mono text-[var(--label2)]">{region.seq[i]} → {scanAlt[i]}</span>
              <span className="sv-mono text-right" style={{ color: heatCss(scan[i]) }}>{scan[i].toFixed(2)}</span>
            </button>
          ))}
        </div>
      )}

      <div className="sv-hr" />
      <div className="flex gap-2">
        <button className="sv-pill" disabled={!edits.length} onClick={() => void store.getState().undo()}>Undo</button>
        <button className="sv-pill" disabled={!edits.length} onClick={() => void store.getState().reset()}>Reset all</button>
      </div>
      <div className="mt-2.5 text-xs text-[var(--label2)]">
        {edits.slice(-4).reverse().map((e, k) => (
          <div key={`${e.position}-${k}`} className="flex h-[22px] items-center justify-between"><span className="sv-mono">{fmt(e.position)}</span><span className="sv-mono">{e.from} → {e.to}</span></div>
        ))}
      </div>
      <div className="mt-3 text-[11px] tracking-[0.01em] text-[var(--label3)]">{engine} · Δ within ±50 bp</div>
    </aside>
  );
}
