"use client";
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useStore } from "zustand";
import type { ExplorerState, ExplorerStore } from "./store";
import type { UIState, UIStore } from "./ui-store";
import type { Sound } from "./sound";
import type { SpliceModel } from "./splice/scorer";
import type { SpliceClient } from "./splice/client";
import type { FrameInfo, SceneEngine } from "@/scene/engine";

export interface IslandOptions { hold?: number; wrap?: boolean }
export interface IslandApi { show(content: ReactNode, opts?: IslandOptions): void; hide(): void }

/** Per-frame values from the 3D engine, delivered without React re-renders. */
export class FrameBus {
  private listeners = new Set<(f: FrameInfo) => void>();
  last: FrameInfo | null = null;
  emit(f: FrameInfo) { this.last = f; for (const l of this.listeners) l(f); }
  subscribe(l: (f: FrameInfo) => void) { this.listeners.add(l); return () => { this.listeners.delete(l); }; }
}

export interface ExplorerCtx {
  store: ExplorerStore;
  ui: UIStore;
  engine: { current: SceneEngine | null };
  frames: FrameBus;
  island: IslandApi;
  sound: Sound;
  scanModel: SpliceModel;
  client: SpliceClient;
}

export const ExplorerContext = createContext<ExplorerCtx | null>(null);
export function useExplorerCtx() {
  const c = useContext(ExplorerContext);
  if (!c) throw new Error("useExplorerCtx outside <Explorer>");
  return c;
}
export function useExplorer<T>(sel: (s: ExplorerState) => T): T { return useStore(useExplorerCtx().store, sel); }
export function useUI<T>(sel: (s: UIState) => T): T { return useStore(useExplorerCtx().ui, sel); }

/** Names that depend on the region being shown (synthetic demo or a real GRCh38 window from the server). */
export function useRegionLabels() {
  const region = useExplorer((s) => s.region);
  const k = region.exons.findIndex(([, e]) => e === region.feature);
  const exon = `${region.gene} exon ${k >= 0 ? k + region.firstExonNumber : ""}`.trim();
  const chromNumber = region.chrom.replace(/^chr/, "");
  return {
    levels: ["Brain", `Chromosome ${chromNumber}`, exon, "Splice landscape"] as const,
    exon,
    band: region.gene === "MAPT" ? "17q21.31 · MAPT" : `${region.chrom} · ${region.gene}`,
    synthetic: region.synthetic,
    sourceNote: region.synthetic ? "synthetic sequence, illustrative coordinates" : region.description ?? "GRCh38",
  };
}

/** Current zoom level (integer), updated only when it changes. */
export function useLevel() {
  const { frames } = useExplorerCtx();
  const [level, setLevel] = useState(0);
  const ref = useRef(0);
  useEffect(() => frames.subscribe((f) => { if (f.level !== ref.current) { ref.current = f.level; setLevel(f.level); } }), [frames]);
  return level;
}
