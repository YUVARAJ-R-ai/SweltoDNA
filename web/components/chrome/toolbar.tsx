"use client";
import { useExplorerCtx, useLevel, useRegionLabels } from "@/lib/explorer-context";

/** Wayfinding: where am I, where can I go. Each crumb jumps the zoom spring to that level. */
export function Toolbar() {
  const { engine, ui } = useExplorerCtx();
  const level = useLevel();
  const LEVELS = useRegionLabels().levels;
  return (
    <nav aria-label="Scale" className="sv-mat sv-regular fixed left-4 top-4 z-[6] flex h-11 items-center gap-1 rounded-[22px] pl-4 pr-2">
      <div className="mr-2.5 flex items-center gap-2 text-sm font-semibold tracking-[-0.01em]">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden>
          <path d="M5 1c0 4 8 4 8 8s-8 4-8 8" /><path d="M13 1c0 4-8 4-8 8s8 4 8 8" /><path d="M6.5 4.5h5M6.5 13.5h5" opacity=".6" />
        </svg>
        Svelto
      </div>
      {LEVELS.map((name, i) => (
        <span key={name} className="flex items-center gap-1">
          {i > 0 && <span className="text-[11px] text-[var(--label3)] max-[900px]:hidden" aria-hidden>›</span>}
          <button
            aria-current={level === i ? "page" : undefined}
            onClick={() => { engine.current?.setZoomTarget(i); ui.getState().set({ radial: null }); }}
            className={`h-[30px] whitespace-nowrap rounded-[15px] px-2.5 text-[13px] transition-colors ${level === i ? "bg-[var(--fill)] font-semibold text-[var(--label)]" : "text-[var(--label2)] hover:bg-[var(--fill2)] hover:text-[var(--label)] max-[900px]:hidden"}`}
          >
            {name}
          </button>
        </span>
      ))}
    </nav>
  );
}
