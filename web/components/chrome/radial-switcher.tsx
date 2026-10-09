"use client";
import { useEffect, useState } from "react";
import { useExplorerCtx, useExplorer, useUI } from "@/lib/explorer-context";
import { BASES, type Base } from "@/lib/splice/sequence";

/** #7 radial base switcher: anchored to the pressed base, kept inside the viewport, edit dispatched in the click handler. */
export function RadialSwitcher({ onPick }: { onPick: (position: number, b: Base) => void }) {
  const { ui } = useExplorerCtx();
  const radial = useUI((s) => s.radial);
  const seq = useExplorer((s) => s.seq);
  const [open, setOpen] = useState(false);
  useEffect(() => { if (!radial) { setOpen(false); return; } const r = requestAnimationFrame(() => setOpen(true)); return () => cancelAnimationFrame(r); }, [radial]);
  if (!radial) return null;
  const x = Math.min(Math.max(radial.x, 90), innerWidth - 90), y = Math.min(Math.max(radial.y, 90), innerHeight - 90);
  return (
    <div role="menu" aria-label={`Change base at position ${radial.position}`}
      className="fixed z-[9] h-[148px] w-[148px] -translate-x-1/2 -translate-y-1/2 transition-[opacity,transform] duration-300 [transition-timing-function:cubic-bezier(.2,.9,.25,1)]"
      style={{ left: x, top: y, opacity: open ? 1 : 0, transform: `translate(-50%,-50%) scale(${open ? 1 : 0.6})` }}>
      <div className="sv-mat sv-regular absolute inset-3.5 rounded-full" />
      {BASES.map((b, k) => {
        const a = -Math.PI / 2 + (k * Math.PI) / 2, cur = seq[radial.position] === b;
        return (
          <button key={b} role="menuitem" disabled={cur} aria-label={`Change to ${b}`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => { onPick(radial.position, b); ui.getState().set({ radial: null }); }}
            className="sv-mono absolute -ml-[21px] -mt-[21px] h-[42px] w-[42px] rounded-full text-[15px] font-bold text-white shadow-[0_4px_14px_rgba(0,0,0,.4)] disabled:opacity-30"
            style={{ left: 74 + Math.cos(a) * 46, top: 74 + Math.sin(a) * 46, background: `var(--${b})` }}>{b}</button>
        );
      })}
    </div>
  );
}
