"use client";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { ExplorerContext, FrameBus, type ExplorerCtx, type IslandApi } from "@/lib/explorer-context";
import { createExplorerStore } from "@/lib/store";
import { createUIStore } from "@/lib/ui-store";
import { Sound } from "@/lib/sound";
import { syntheticRegion, type Base } from "@/lib/splice/sequence";
import { SpliceModel, impactTier } from "@/lib/splice/scorer";
import { LocalHeuristicClient, WebSocketClient } from "@/lib/splice/client";
import { runTour } from "@/lib/tour";
import { SceneEngine } from "@/scene/engine";
import { WINDOW } from "@/lib/store";
import { Toolbar } from "./chrome/toolbar";
import { Tools } from "./chrome/tools";
import { AboutPopover } from "./chrome/about-popover";
import { PerformanceCard } from "./chrome/performance-card";
import { DepthScrubber } from "./chrome/depth-scrubber";
import { WorldLabels } from "./chrome/world-labels";
import { Inspector } from "./chrome/inspector";
import { SequenceRibbon } from "./chrome/sequence-ribbon";
import { RadialSwitcher } from "./chrome/radial-switcher";
import { SnapshotSheet } from "./chrome/snapshot-sheet";
import { DynamicIsland, Dot } from "./chrome/dynamic-island";
import { Hint } from "./chrome/hint";

const TIER_COLOR = { High: "var(--red)", Moderate: "var(--orange)", Low: "var(--yellow)", Minimal: "var(--label2)" } as const;

export default function Explorer() {
  const islandRef = useRef<IslandApi>(null);
  const host = useRef<HTMLDivElement>(null);
  const tourAbort = useRef<AbortController | null>(null);

  const ctx = useMemo<ExplorerCtx>(() => {
    const region = syntheticRegion();
    const ws = process.env.NEXT_PUBLIC_SPLICE_WS;
    const client = ws ? new WebSocketClient(ws, crypto.randomUUID()) : new LocalHeuristicClient(region);
    return {
      store: createExplorerStore(region, client), ui: createUIStore(), engine: { current: null }, frames: new FrameBus(), sound: new Sound(),
      scanModel: new SpliceModel(region),
      island: { show: (n, o) => islandRef.current?.show(n, o), hide: () => islandRef.current?.hide() },
    };
  }, []);
  const { store, ui, engine, frames, island, sound, scanModel } = ctx;

  const mutate = async (i: number, b: Base) => {
    if (store.getState().seq[i] === b) return;
    await store.getState().mutate(i, b);
    const s = store.getState();
    if (s.error) { island.show(<><Dot color="var(--red)" />{s.error}</>, { hold: 3000 }); return; }
    const d = s.deltaAt(i), max = d?.max ?? 0, tier = impactTier(max);
    if (!ui.getState().tourOn) island.show(<><Dot color={TIER_COLOR[tier]} />{tier} impact<span className="sv-mono text-[var(--label2)]">Δ {max.toFixed(2)}</span></>, { hold: 2200 });
    sound.edit(max);
  };

  const scan = () => {
    const e = engine.current; if (!e || e.scanning) return;
    const w0 = store.getState().windowStart, positions = Array.from({ length: WINDOW }, (_, k) => w0 + k);
    ui.getState().set({ scanning: true });
    if (!ui.getState().tourOn) island.show(<><Dot color="var(--blue)" />Scanning {WINDOW * 3} variants</>, { hold: 0 });
    let k = 0;
    e.runScan(positions, (i) => { const r = scanModel.scanPosition(i); store.getState().setScan(i, r.best, r.alt); if (k++ % 3 === 0) sound.scanTick(r.best); }, () => {
      const sc = store.getState().scan, hi = positions.filter((i) => sc[i] > 0.8).length, mod = positions.filter((i) => sc[i] > 0.5 && sc[i] <= 0.8).length;
      ui.getState().set({ scanning: false });
      island.show(<><Dot color="var(--red)" />{WINDOW * 3} variants scored<span className="text-[var(--label2)]">{hi} high · {mod} moderate</span></>, { hold: 3200 });
      sound.chord();
    });
  };

  const openSnapshot = () => {
    const e = engine.current; if (!e) return;
    if (ui.getState().snapshot) { ui.getState().set({ snapshot: null }); return; }
    sound.shutter();
    ui.getState().set({ radial: null, snapshot: e.capture() });
  };

  const toggleTour = () => {
    if (tourAbort.current) { tourAbort.current.abort(); return; }
    const ac = (tourAbort.current = new AbortController());
    ui.getState().set({ tourOn: true, hintGone: true, radial: null });
    runTour(ctx, ac.signal, { mutate, scan }).catch(() => {}).finally(() => {
      if (tourAbort.current === ac) tourAbort.current = null;
      ui.getState().set({ tourOn: false });
      if (ac.signal.aborted) island.hide();
    });
  };

  // engine lifecycle
  useEffect(() => {
    const e = new SceneEngine(host.current!, store);
    engine.current = e;
    e.onFrame = (f) => frames.emit(f);
    e.onRungTap = (i, x, y) => { store.getState().select(i); ui.getState().set({ radial: { position: i, x, y } }); sound.select(); };
    e.onInteract = () => ui.getState().set({ hintGone: true });
    void store.getState().init();
    const hash = location.hash.match(/z=(\d)/);                         // presenter/test shortcut: #z=2, #z=2&mut
    if (hash) {
      e.zoom.x = e.zoomTarget = +hash[1]; ui.getState().set({ hintGone: true });
      if (location.hash.includes("mut")) setTimeout(() => void mutate(store.getState().region.feature + 1, "A"), 300);
    }
    return () => { e.dispose(); engine.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // global input: unlock audio, let any input interrupt the tour, keyboard shortcuts
  useEffect(() => {
    const down = (ev: PointerEvent) => { sound.unlock(); if (!(ev.target as HTMLElement).closest("[aria-label='Guided tour']")) tourAbort.current?.abort(); };
    const wheel = () => tourAbort.current?.abort();
    const key = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).closest("input,textarea")) return;
      const k = ev.key.toLowerCase(), u = ui.getState(), s = store.getState();
      if (k === "p" && !ev.metaKey && !ev.ctrlKey) { toggleTour(); return; }
      tourAbort.current?.abort();
      if (u.snapshot) { if (k === "escape") u.set({ snapshot: null }); return; }
      if ((ev.metaKey || ev.ctrlKey) && k === "z") { ev.preventDefault(); void s.undo(); return; }
      if (k === "escape") { u.set({ radial: null, aboutOpen: false }); return; }
      if (k === "s" && !ev.metaKey && !ev.ctrlKey) { openSnapshot(); return; }
      if (ev.key >= "1" && ev.key <= "4") { engine.current?.setZoomTarget(+ev.key - 1); return; }
      if (k === "+" || k === "=") { engine.current?.setZoomTarget(Math.round(engine.current.zoomTarget) + 1); return; }
      if (k === "-") { engine.current?.setZoomTarget(Math.round(engine.current.zoomTarget) - 1); return; }
      if ((frames.last?.level ?? 0) < 2) return;
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { ev.preventDefault(); s.select(s.selected + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 10 : 1)); sound.select(); return; }
      const b = ev.key.toUpperCase();
      if (b.length === 1 && "ACGT".includes(b) && !ev.metaKey && !ev.ctrlKey) { u.set({ radial: null }); void mutate(s.selected, b as Base); }
    };
    addEventListener("pointerdown", down, true); addEventListener("wheel", wheel, { capture: true, passive: true }); addEventListener("keydown", key);
    return () => { removeEventListener("pointerdown", down, true); removeEventListener("wheel", wheel, true); removeEventListener("keydown", key); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <ExplorerContext.Provider value={ctx}>
      <main className="sv-root">
        <div ref={host} className="fixed inset-0" data-testid="scene" />
        <Chrome>
          <Toolbar />
          <Tools onTour={toggleTour} onSnapshot={openSnapshot} />
          <AboutPopover />
          <DepthScrubber />
          <WorldLabels />
          <Inspector onMutate={(i, b) => void mutate(i, b)} onScan={scan} />
          <PerformanceCard />
          <SequenceRibbon onPick={(i, x, y) => { ui.getState().set({ radial: { position: i, x, y } }); sound.select(); }} />
          <RadialSwitcher onPick={(i, b) => void mutate(i, b)} />
          <SnapshotSheet />
          <Hint />
        </Chrome>
        <DynamicIsland ref={islandRef} />
      </main>
    </ExplorerContext.Provider>
  );
}

const Chrome = ({ children }: { children: ReactNode }) => <>{children}</>;
