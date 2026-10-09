"use client";
import { useEffect, useRef, useState } from "react";
import { useExplorerCtx, useLevel, useRegionLabels } from "@/lib/explorer-context";

/** Direct manipulation of zoom: grab offset respected, rubber-banded ends, release velocity handed to the spring. */
export function DepthScrubber() {
  const { frames, engine } = useExplorerCtx();
  const level = useLevel();
  const LEVELS = useRegionLabels().levels;
  const root = useRef<HTMLDivElement>(null), thumb = useRef<HTMLDivElement>(null), fill = useRef<HTMLDivElement>(null), label = useRef<HTMLDivElement>(null);
  const drag = useRef<{ top: number; H: number; off: number; hist: [number, number][] } | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => frames.subscribe((f) => {
    const p = `${(Math.max(-0.2, Math.min(3.2, f.z)) / 3) * 100}%`;
    if (thumb.current) thumb.current.style.top = p;
    if (label.current) label.current.style.top = p;
    if (fill.current) fill.current.style.height = `${(Math.max(0, Math.min(3, f.z)) / 3) * 100}%`;
  }), [frames]);

  return (
    <div ref={root} role="slider" tabIndex={0} aria-label="Zoom level" aria-valuemin={0} aria-valuemax={3} aria-valuenow={level} aria-valuetext={LEVELS[level]}
      className="group fixed left-6 top-1/2 z-[5] h-[232px] w-11 -translate-y-1/2 cursor-grab touch-none max-[900px]:hidden"
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); engine.current?.setZoomTarget(level + 1); }
        if (e.key === "ArrowUp") { e.preventDefault(); engine.current?.setZoomTarget(level - 1); }
      }}
      onPointerDown={(e) => {
        const r = root.current!.getBoundingClientRect(), e0 = engine.current; if (!e0) return;
        root.current!.setPointerCapture(e.pointerId); setDragging(true);
        const thumbY = (e0.zoom.x / 3) * r.height, y = e.clientY - r.top, onThumb = Math.abs(y - thumbY) < 16;
        drag.current = { top: r.top, H: r.height, off: onThumb ? y - thumbY : 0, hist: [] };
        if (!onThumb) e0.setZoomTarget((y / r.height) * 3);
      }}
      onPointerMove={(e) => {
        const d = drag.current; if (!d) return;
        const z = ((e.clientY - d.top - d.off) / d.H) * 3;
        engine.current?.scrubTo(z);
        d.hist.push([z, performance.now()]); if (d.hist.length > 6) d.hist.shift();
      }}
      onPointerUp={() => {
        const d = drag.current; drag.current = null; setDragging(false); if (!d) return;
        const h = d.hist, v = h.length > 1 ? ((h[h.length - 1][0] - h[0][0]) / Math.max(16, h[h.length - 1][1] - h[0][1])) * 1000 : 0;
        engine.current?.scrubEnd(v);
      }}>
      <div className="absolute bottom-0 left-5 top-0 w-1 rounded-sm bg-[var(--fill)]" />
      <div ref={fill} className="absolute left-5 top-0 w-1 rounded-sm bg-[var(--label2)]" />
      {[0, 1, 2, 3].map((i) => <div key={i} className="absolute left-[18px] -mt-1 h-2 w-2 rounded bg-[var(--label3)]" style={{ top: `${(i / 3) * 100}%` }} />)}
      <div ref={thumb} className="absolute left-[11px] -mt-[11px] h-[22px] w-[22px] rounded-full bg-[#f5f5f7] shadow-[0_2px_10px_rgba(0,0,0,.5)]" />
      <div ref={label} className={`absolute left-11 -mt-[9px] whitespace-nowrap text-xs transition-opacity group-hover:opacity-100 ${dragging ? "opacity-100" : "opacity-0"}`}>{LEVELS[level]}</div>
    </div>
  );
}
