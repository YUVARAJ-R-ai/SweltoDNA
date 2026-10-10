"use client";
import { useExplorerCtx, useUI } from "@/lib/explorer-context";

export function Tools({ onTour, onSnapshot }: { onTour: () => void; onSnapshot: () => void }) {
  const { ui, sound } = useExplorerCtx();
  const { tourOn, soundOn, perfOpen, aboutOpen, snapshot } = useUI((s) => s);
  const set = ui.getState().set;
  return (
    <div className="fixed right-4 top-4 z-[6] flex gap-2">
      <button className="sv-round sv-mat sv-regular" aria-pressed={tourOn} aria-label="Guided tour" title="Guided tour (P)" onClick={onTour}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor" aria-hidden><path d="M5 3.2v11.6c0 .6.7 1 1.2.6l8.6-5.8a.75.75 0 0 0 0-1.2L6.2 2.6C5.7 2.2 5 2.6 5 3.2z" /></svg>
      </button>
      <button className="sv-round sv-mat sv-regular" aria-pressed={soundOn} aria-label="Sound" title="Sound"
        onClick={() => { sound.enabled = !soundOn; set({ soundOn: !soundOn }); }}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden>
          <path d="M3.5 7.5h3l4-3.5v12l-4-3.5h-3z" fill="currentColor" stroke="none" />
          <path d="M13.5 7.2a4 4 0 0 1 0 5.6" opacity={soundOn ? 1 : 0} /><path d="M15.8 5a7 7 0 0 1 0 10" opacity={soundOn ? 1 : 0} />
        </svg>
      </button>
      <button className="sv-round sv-mat sv-regular" aria-pressed={!!snapshot} aria-label="Snapshot" title="Snapshot (S)" onClick={onSnapshot}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" aria-hidden>
          <path d="M2.8 6.8c0-.9.7-1.6 1.6-1.6h2l1.2-1.8h4.8l1.2 1.8h2c.9 0 1.6.7 1.6 1.6v7.6c0 .9-.7 1.6-1.6 1.6H4.4c-.9 0-1.6-.7-1.6-1.6z" /><circle cx="10" cy="10.6" r="3" />
        </svg>
      </button>
      <button className="sv-round sv-mat sv-regular" aria-pressed={perfOpen} aria-label="Performance" title="Performance" onClick={() => set({ perfOpen: !perfOpen })}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden><path d="M3.5 14a7 7 0 1 1 13 0" /><path d="M10 13.5l3.5-4.5" /></svg>
      </button>
      <button className="sv-round sv-mat sv-regular" aria-pressed={aboutOpen} aria-label="About this demo" title="About this demo" onClick={() => set({ aboutOpen: !aboutOpen })}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden><circle cx="10" cy="10" r="7.5" /><path d="M10 9v5" /><circle cx="10" cy="6.4" r=".6" fill="currentColor" /></svg>
      </button>
    </div>
  );
}
