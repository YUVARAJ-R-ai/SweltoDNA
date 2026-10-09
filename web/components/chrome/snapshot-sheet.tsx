"use client";
import { useState } from "react";
import { useExplorerCtx, useRegionLabels, useUI } from "@/lib/explorer-context";
import { DitherShader, ditherToCanvas, fitImageData, type ColorMode, type DitheringMode } from "@/components/ui/dither-shader";
import { describeDelta, impactTier } from "@/lib/splice/scorer";
import { Dot } from "./dynamic-island";

const CW = 372, CH = 465;
const TONES = {
  mono: { label: "Mono", colorMode: "grayscale" as ColorMode, paper: "#000000", ink: "#ffffff", primary: "#000000", secondary: "#ffffff" },
  paper: { label: "Paper", colorMode: "duotone" as ColorMode, paper: "#efeae0", ink: "#1c1c1e", primary: "#efeae0", secondary: "#1c1c1e" },
  blueprint: { label: "Blueprint", colorMode: "duotone" as ColorMode, paper: "#0b2a5b", ink: "#cfe3ff", primary: "#0b2a5b", secondary: "#cfe3ff" },
  colour: { label: "Colour", colorMode: "original" as ColorMode, paper: "#000000", ink: "#ffffff", primary: "#000000", secondary: "#ffffff" },
};
type Tone = keyof typeof TONES;
const MODES: DitheringMode[] = ["bayer", "halftone", "crosshatch", "noise"];

function wrap(g: CanvasRenderingContext2D, text: string, maxW: number) {
  const out: string[] = []; let cur = "";
  for (const w of text.split(" ")) { const t = cur ? `${cur} ${w}` : w; if (g.measureText(t).width > maxW && cur) { out.push(cur); cur = w; } else cur = t; }
  if (cur) out.push(cur); return out;
}

/** 1-bit print of the current view (modal: dims the scene; grows from the camera button). */
export function SnapshotSheet() {
  const { ui, store, island, sound } = useExplorerCtx();
  const url = useUI((s) => s.snapshot);
  const labels = useRegionLabels();
  const [mode, setMode] = useState<DitheringMode>("bayer");
  const [tone, setTone] = useState<Tone>("paper");
  const [grain, setGrain] = useState(2);
  const reduced = typeof window !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const T = TONES[tone];

  const s = store.getState(), last = s.edits[s.edits.length - 1], i = last ? last.position : s.selected;
  const changed = s.seq[i] !== s.region.seq[i], d = changed ? s.deltaAt(i) : null;
  const card = {
    title: changed ? `${s.region.seq[i]} → ${s.seq[i]}` : labels.exon,
    sub: `${labels.exon} · ${s.region.chrom}:${(s.region.coord0 + i).toLocaleString("en-US")}`,
    line: d ? `${impactTier(d.max)} impact. ${describeDelta(d, i)}` : "A synthetic splice-site explorer.",
    foot: `Svelto · ${labels.synthetic ? "synthetic sequence" : s.region.chrom + " GRCh38"} · ${s.engine.toLowerCase()}`,
  };
  const params = { gridSize: grain, ditherMode: mode, colorMode: T.colorMode, primaryColor: T.primary, secondaryColor: T.secondary, contrast: 1.35, brightness: 0.04, threshold: 0.5, backgroundColor: T.paper };
  const close = () => ui.getState().set({ snapshot: null });

  const save = () => {
    if (!url) return;
    const img = new Image();
    img.onload = () => {
      const data = fitImageData(img, CW, CH); if (!data) return;
      const base = document.createElement("canvas"); base.width = CW; base.height = CH;
      ditherToCanvas(base.getContext("2d")!, data, CW, CH, params, 0);
      const K = 3, W = CW * K, H = CH * K, c = document.createElement("canvas"); c.width = W; c.height = H;
      const g = c.getContext("2d")!; g.imageSmoothingEnabled = false; g.drawImage(base, 0, 0, W, H);   // nearest-neighbour keeps dots crisp
      const pad = 18 * K, sans = "-apple-system, BlinkMacSystemFont, system-ui, sans-serif";
      g.font = `400 ${13 * K}px ${sans}`; const lines = wrap(g, card.line, W - pad * 2);
      const bandH = (16 + 35.5 + 25 + 10 + 14 + 14) * K + lines.length * 17.5 * K;
      g.fillStyle = T.paper; g.fillRect(0, H - bandH, W, bandH); g.fillStyle = T.ink; g.textBaseline = "top";
      let y = H - bandH + 16 * K;
      g.font = `600 ${30 * K}px "SF Mono", ui-monospace, Menlo, monospace`; g.fillText(card.title, pad, y); y += 35.5 * K;
      g.globalAlpha = 0.75; g.font = `400 ${12 * K}px ${sans}`; g.fillText(card.sub, pad, y); y += 25 * K;
      g.globalAlpha = 1; g.font = `400 ${13 * K}px ${sans}`; for (const l of lines) { g.fillText(l, pad, y); y += 17.5 * K; }
      y += 10 * K; g.globalAlpha = 0.55; g.font = `400 ${10 * K}px ${sans}`; g.fillText(card.foot, pad, y); g.globalAlpha = 1;
      c.toBlob((b) => {
        if (!b) return;
        const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `svelto-${s.region.coord0 + i}-${mode}-${tone}.png`;
        a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        island.show(<><Dot color="var(--green)" />Image saved</>, { hold: 1800 }); sound.tone(880, 0.12, "sine", 0.03); sound.tone(1320, 0.18, "sine", 0.025, 0.06);
      }, "image/png");
    };
    img.src = url;
  };

  const seg = (on: boolean) => `h-[30px] rounded-[9px] text-xs font-medium ${on ? "bg-white/15 shadow-[0_1px_3px_rgba(0,0,0,.35)]" : ""}`;
  return (
    <>
      <div className={`fixed inset-0 z-[9] bg-black/45 transition-opacity duration-300 ${url ? "opacity-100" : "pointer-events-none opacity-0"}`} onClick={close} />
      <section role="dialog" aria-modal="true" aria-label="Snapshot" data-off={!url} data-scroll-own
        className="sv-mat sv-thick sv-panel fixed right-4 top-[68px] z-10 max-h-[calc(100vh-84px)] w-[404px] origin-[calc(100%-214px)_-12px] overflow-y-auto overflow-x-hidden rounded-3xl p-4 data-[off=true]:scale-90">
        <div className="mb-3 flex items-center justify-between"><h3 className="text-[15px] font-semibold tracking-[-0.01em]">Snapshot</h3><span className="text-xs text-[var(--label2)]">1-bit print of this view</span></div>
        <div className="relative h-[465px] w-[372px] overflow-hidden rounded-[14px] shadow-[0_0_0_.5px_rgba(255,255,255,.14)]">
          {url && <DitherShader src={url} {...params} animated={mode === "noise" && !reduced} className="absolute inset-0" />}
          <div className="absolute inset-x-0 bottom-0 px-[18px] pb-3.5 pt-4" style={{ background: T.paper, color: T.ink }}>
            <div className="sv-mono text-[30px] font-semibold leading-[1.05] tracking-[-0.02em]">{card.title}</div>
            <div className="mt-1 text-xs opacity-75">{card.sub}</div>
            <div className="mt-2 text-[13px] leading-[1.35]">{card.line}</div>
            <div className="mt-2.5 text-[10px] tracking-[0.01em] opacity-55">{card.foot}</div>
          </div>
        </div>
        <div className="sv-seg mt-3" role="group" aria-label="Pattern">
          {MODES.map((m) => <button key={m} aria-pressed={mode === m} className={seg(mode === m)} onClick={() => setMode(m)}>{m[0].toUpperCase() + m.slice(1)}</button>)}
        </div>
        <div className="sv-seg mt-2" role="group" aria-label="Tone">
          {(Object.keys(TONES) as Tone[]).map((t) => <button key={t} aria-pressed={tone === t} className={seg(tone === t)} onClick={() => setTone(t)}>{TONES[t].label}</button>)}
        </div>
        <label className="mt-3 flex items-center gap-3 text-xs text-[var(--label2)]">Grain
          <input type="range" min={1} max={6} step={1} value={grain} onChange={(e) => setGrain(+e.target.value)} className="flex-1 accent-[var(--blue)]" />
        </label>
        <div className="mt-3.5 flex justify-end gap-2">
          <button className="sv-pill" onClick={close}>Done</button>
          <button className="sv-pill sv-primary" onClick={save}>Save image</button>
        </div>
      </section>
    </>
  );
}
