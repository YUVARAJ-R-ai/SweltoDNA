"use client";
import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { IslandApi, IslandOptions } from "@/lib/explorer-context";
import { stepSpring } from "@/lib/springs";

interface Item { node: ReactNode; wrap: boolean; id: number }

/** One black pill that morphs (slightly under-damped spring) for results, progress, and tour captions. */
export const DynamicIsland = forwardRef<IslandApi>(function DynamicIsland(_, ref) {
  const outer = useRef<HTMLDivElement>(null), measure = useRef<HTMLDivElement>(null);
  const [next, setNext] = useState<Item | null>(null);
  const [cur, setCur] = useState<Item | null>(null);
  const [innerOn, setInnerOn] = useState(false);
  const target = useRef({ w: 0, h: 44 }), shown = useRef(false), holdT = useRef(0), swapT = useRef(0);

  const hide = () => { setInnerOn(false); target.current = { w: 0, h: 44 }; shown.current = false; };
  useImperativeHandle(ref, () => ({
    show(node: ReactNode, { hold = 2400, wrap = false }: IslandOptions = {}) {
      setNext({ node, wrap, id: Math.random() });
      clearTimeout(holdT.current);
      if (hold > 0) holdT.current = window.setTimeout(hide, hold);
    },
    hide,
  }), []);

  useLayoutEffect(() => {
    if (!next || !measure.current) return;
    target.current = { w: measure.current.offsetWidth, h: Math.max(44, measure.current.offsetHeight) };
    clearTimeout(swapT.current);
    if (shown.current) { setInnerOn(false); swapT.current = window.setTimeout(() => { setCur(next); setInnerOn(true); }, 120); }
    else { setCur(next); setInnerOn(true); }
    shown.current = true;
  }, [next]);

  useEffect(() => {
    const w = { x: 0, v: 0 }, h = { x: 44, v: 0 }; let last = performance.now(), raf = 0;
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 1 / 30); last = now;
      stepSpring(w, target.current.w, dt, 0.42, 0.74); stepSpring(h, target.current.h, dt, 0.42, 0.86);
      const el = outer.current;
      if (el) { el.style.width = `${Math.max(0, w.x)}px`; el.style.height = `${Math.max(0, h.x)}px`; el.style.opacity = shown.current ? "1" : `${Math.min(1, w.x / 60)}`; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const inner = (wrap: boolean) =>
    `flex items-center gap-2.5 font-medium tracking-[-0.005em] ${wrap ? "w-[460px] whitespace-normal px-5 py-[11px] text-sm leading-[1.4]" : "w-max max-w-[460px] whitespace-nowrap px-[18px] text-[13px]"} min-h-11`;
  return (
    <>
      <div ref={outer} role="status" aria-live="polite" data-testid="island"
        className="pointer-events-none fixed left-1/2 top-4 z-[8] h-11 w-0 -translate-x-1/2 overflow-hidden rounded-[22px] bg-black opacity-0 shadow-[0_0_0_.5px_rgba(255,255,255,.16),0_10px_30px_rgba(0,0,0,.55)] max-[1360px]:top-[68px]">
        {cur && <div className={`absolute left-0 top-0 transition-opacity duration-200 ${inner(cur.wrap)}`} style={{ opacity: innerOn ? 1 : 0 }}>{cur.node}</div>}
      </div>
      <div ref={measure} aria-hidden className={`invisible fixed -left-[9999px] top-0 ${inner(next?.wrap ?? false)}`}>{next?.node}</div>
    </>
  );
});

export const Dot = ({ color }: { color: string }) => <span className="inline-block h-2 w-2 flex-none rounded-full" style={{ background: color }} />;
