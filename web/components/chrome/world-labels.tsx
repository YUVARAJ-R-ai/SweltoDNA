"use client";
import { useEffect, useRef } from "react";
import { useExplorerCtx, useExplorer } from "@/lib/explorer-context";
import type { ScreenAnchor } from "@/scene/engine";

/** DOM labels pinned to 3D anchors; positions come from the frame bus, not React state. */
export function WorldLabels() {
  const { frames, engine } = useExplorerCtx();
  const delta = useExplorer((s) => s.delta);
  const hot = useRef<HTMLButtonElement>(null), band = useRef<HTMLButtonElement>(null);
  const loss = useRef<HTMLDivElement>(null), gain = useRef<HTMLDivElement>(null);
  useEffect(() => frames.subscribe((f) => {
    const place = (el: HTMLElement | null, a: ScreenAnchor) => {
      if (!el) return;
      el.style.opacity = a.visible ? "1" : "0"; el.style.pointerEvents = a.visible ? "auto" : "none";
      if (a.visible) { el.style.left = `${a.x}px`; el.style.top = `${a.y}px`; }
    };
    place(hot.current, f.anchors.hot); place(band.current, f.anchors.band); place(loss.current, f.anchors.loss); place(gain.current, f.anchors.gain);
  }), [frames]);
  const lossName = delta && delta.acceptorLoss > delta.donorLoss ? "Acceptor lost" : "Donor lost";
  const lossVal = delta ? Math.max(delta.donorLoss, delta.acceptorLoss) : 0;
  const gainName = delta && delta.acceptorGain > delta.donorGain ? "Cryptic acceptor" : "Cryptic donor";
  const gainVal = delta ? Math.max(delta.donorGain, delta.acceptorGain) : 0;
  return (
    <>
      <button ref={hot} className="sv-tag3d sv-mat sv-regular cursor-pointer" style={{ opacity: 0 }} onClick={() => engine.current?.setZoomTarget(1)}>MAPT</button>
      <button ref={band} className="sv-tag3d sv-mat sv-regular cursor-pointer" style={{ opacity: 0 }} onClick={() => engine.current?.setZoomTarget(2)}>17q21.31 · MAPT</button>
      <div ref={loss} className="sv-tag3d sv-mat sv-regular" style={{ opacity: 0, color: "var(--red)" }}>{lossName} <span className="sv-mono">−{lossVal.toFixed(2)}</span></div>
      <div ref={gain} className="sv-tag3d sv-mat sv-regular" style={{ opacity: 0, color: "var(--orange)" }}>{gainName} <span className="sv-mono">+{gainVal.toFixed(2)}</span></div>
    </>
  );
}
