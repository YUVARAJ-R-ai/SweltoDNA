"use client";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/*
 * Vendored from the provided DitherShader component, with two fixes from review:
 *  1. A new `src` is always loaded (the original reused the previous image whenever it had finished loading).
 *  2. Parsed colors are memoized, so unrelated parent re-renders no longer re-dither or restart the animation.
 * The pixel algorithm lives in `ditherToCanvas` so exports can reuse it exactly.
 */

export type DitheringMode = "bayer" | "halftone" | "noise" | "crosshatch";
export type ColorMode = "original" | "grayscale" | "duotone" | "custom";

export interface DitherParams {
  gridSize?: number;
  ditherMode?: DitheringMode;
  colorMode?: ColorMode;
  invert?: boolean;
  pixelRatio?: number;
  primaryColor?: string;
  secondaryColor?: string;
  customPalette?: string[];
  brightness?: number;
  contrast?: number;
  backgroundColor?: string;
  threshold?: number;
}

interface DitherShaderProps extends DitherParams {
  /** Source image URL */
  src: string;
  /** Object fit behavior */
  objectFit?: "cover" | "contain" | "fill" | "none";
  /** Enable animation effect */
  animated?: boolean;
  /** Animation speed (lower = slower) */
  animationSpeed?: number;
  /** Additional CSS classes for the container (use this to set size via Tailwind) */
  className?: string;
}

const BAYER_MATRIX_4x4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
const BAYER_MATRIX_8x8 = [
  [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
  [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25], [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
];

type RGB = [number, number, number];
export function parseColor(color: string): RGB {
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3) return [parseInt(hex[0] + hex[0], 16), parseInt(hex[1] + hex[1], 16), parseInt(hex[2] + hex[2], 16)];
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
  }
  const match = color.match(/rgb\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)/i);
  return match ? [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])] : [0, 0, 0];
}
const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

/** Fit an image into W×H the way CSS object-fit would, and return its pixels. */
export function fitImageData(img: HTMLImageElement, W: number, H: number, objectFit: DitherShaderProps["objectFit"] = "cover"): ImageData | null {
  const off = document.createElement("canvas"); off.width = W; off.height = H;
  const g = off.getContext("2d"); if (!g) return null;
  const iw = img.naturalWidth || W, ih = img.naturalHeight || H;
  let dw = W, dh = H, dx = 0, dy = 0;
  if (objectFit === "cover" || objectFit === "contain") {
    const s = objectFit === "cover" ? Math.max(W / iw, H / ih) : Math.min(W / iw, H / ih);
    dw = Math.ceil(iw * s); dh = Math.ceil(ih * s); dx = Math.floor((W - dw) / 2); dy = Math.floor((H - dh) / 2);
  } else if (objectFit === "none") { dw = iw; dh = ih; dx = Math.floor((W - dw) / 2); dy = Math.floor((H - dh) / 2); }
  g.drawImage(img, dx, dy, dw, dh);
  try { return g.getImageData(0, 0, W, H); } catch { console.error("Could not get image data. CORS issue?"); return null; }
}

/** The dithering pass, unchanged from the original component. */
export function ditherToCanvas(ctx: CanvasRenderingContext2D, source: ImageData, W: number, H: number, p: DitherParams, time = 0,
  parsed?: { primary: RGB; secondary: RGB; palette: RGB[] }) {
  const { gridSize = 4, ditherMode = "bayer", colorMode = "original", invert = false, pixelRatio = 1, brightness = 0, contrast = 1,
    backgroundColor = "transparent", threshold = 0.5 } = p;
  const primary = parsed?.primary ?? parseColor(p.primaryColor ?? "#000000");
  const secondary = parsed?.secondary ?? parseColor(p.secondaryColor ?? "#ffffff");
  const palette = parsed?.palette ?? (p.customPalette ?? ["#000000", "#ffffff"]).map(parseColor);
  if (backgroundColor !== "transparent") { ctx.fillStyle = backgroundColor; ctx.fillRect(0, 0, W, H); } else ctx.clearRect(0, 0, W, H);
  const d = source.data, sw = source.width, sh = source.height;
  const px = Math.max(1, Math.floor(gridSize * pixelRatio));
  const ms = gridSize <= 4 ? 4 : 8, bm = gridSize <= 4 ? BAYER_MATRIX_4x4 : BAYER_MATRIX_8x8, mscale = ms === 4 ? 16 : 64;
  for (let y = 0; y < H; y += px) {
    for (let x = 0; x < W; x += px) {
      const si = (Math.floor((y / H) * sh) * sw + Math.floor((x / W) * sw)) * 4;
      if ((d[si + 3] || 0) < 10) continue;
      const r = clamp((d[si] - 128) * contrast + 128 + brightness * 255, 0, 255);
      const g = clamp((d[si + 1] - 128) * contrast + 128 + brightness * 255, 0, 255);
      const b = clamp((d[si + 2] - 128) * contrast + 128 + brightness * 255, 0, 255);
      const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      const mx = Math.floor(x / gridSize) % ms, my = Math.floor(y / gridSize) % ms;
      let th: number;
      switch (ditherMode) {
        case "halftone": { const a = Math.PI / 4, sc = gridSize * 2; th = (Math.sin((x * Math.cos(a) + y * Math.sin(a)) / sc) + Math.sin((-x * Math.sin(a) + y * Math.cos(a)) / sc) + 2) / 4; break; }
        case "noise": { const n = Math.sin(x * 12.9898 + y * 78.233 + time * 100) * 43758.5453; th = n - Math.floor(n); break; }
        case "crosshatch": th = (((x + y) % (gridSize * 2) < gridSize ? 1 : 0) + ((x - y + gridSize * 4) % (gridSize * 2) < gridSize ? 1 : 0)) / 2; break;
        default: th = bm[my][mx] / mscale;
      }
      th = th * (1 - threshold) + threshold * 0.5;
      let o: RGB;
      if (colorMode === "grayscale") o = lum < th ? [0, 0, 0] : [255, 255, 255];
      else if (colorMode === "duotone") o = lum < th ? primary : secondary;
      else if (colorMode === "custom") {
        if (palette.length === 2) o = lum < th ? palette[0] : palette[1];
        else o = palette[Math.floor(clamp(lum + (th - 0.5) * 0.5, 0, 1) * (palette.length - 1))];
      } else {
        const da = (th - 0.5) * 64, L = 255 / 4;
        o = [Math.round(clamp(r + da, 0, 255) / L) * L, Math.round(clamp(g + da, 0, 255) / L) * L, Math.round(clamp(b + da, 0, 255) / L) * L];
      }
      if (invert) o = [255 - o[0], 255 - o[1], 255 - o[2]];
      ctx.fillStyle = `rgb(${o[0]}, ${o[1]}, ${o[2]})`;
      ctx.fillRect(x, y, px, px);
    }
  }
}

export const DitherShader: React.FC<DitherShaderProps> = ({
  src, gridSize = 4, ditherMode = "bayer", colorMode = "original", invert = false, pixelRatio = 1,
  primaryColor = "#000000", secondaryColor = "#ffffff", customPalette, brightness = 0, contrast = 1,
  backgroundColor = "transparent", objectFit = "cover", threshold = 0.5, animated = false, animationSpeed = 0.02, className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const timeRef = useRef(0);
  const imageRef = useRef<{ src: string; img: HTMLImageElement } | null>(null);
  const imageDataRef = useRef<ImageData | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Fix 2: stable parsed colors keyed on their inputs
  const paletteKey = (customPalette ?? ["#000000", "#ffffff"]).join(",");
  const parsed = useMemo(
    () => ({ primary: parseColor(primaryColor), secondary: parseColor(secondaryColor), palette: paletteKey.split(",").map(parseColor) }),
    [primaryColor, secondaryColor, paletteKey],
  );

  const applyDithering = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, time = 0) => {
      if (!imageDataRef.current) return;
      ditherToCanvas(ctx, imageDataRef.current, w, h,
        { gridSize, ditherMode, colorMode, invert, pixelRatio, brightness, contrast, backgroundColor, threshold }, time, parsed);
    },
    [gridSize, ditherMode, colorMode, invert, pixelRatio, brightness, contrast, backgroundColor, threshold, parsed],
  );

  useEffect(() => {
    const el = containerRef.current; if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) { const { width, height } = e.contentRect; if (width > 0 && height > 0) setDimensions({ width, height }); }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0 || dimensions.height === 0) return;
    let cancelled = false;
    const process = (img: HTMLImageElement) => {
      if (cancelled) return;
      const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
      const { width: w, height: h } = dimensions;
      canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
      const ctx = canvas.getContext("2d"); if (!ctx) return;
      ctx.resetTransform(); ctx.scale(dpr, dpr);
      imageDataRef.current = fitImageData(img, w, h, objectFit);
      if (!imageDataRef.current) return;
      applyDithering(ctx, w, h, timeRef.current);
      if (animated) {
        const tick = () => { if (cancelled) return; timeRef.current += animationSpeed; applyDithering(ctx, w, h, timeRef.current); animationRef.current = requestAnimationFrame(tick); };
        animationRef.current = requestAnimationFrame(tick);
      }
    };
    // Fix 1: only reuse the cached image when it is the same source
    if (imageRef.current && imageRef.current.src === src && imageRef.current.img.complete) process(imageRef.current.img);
    else {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { if (cancelled) return; imageRef.current = { src, img }; process(img); };
      img.onerror = () => console.error("Failed to load image for DitherShader:", src);
      img.src = src;
    }
    return () => { cancelled = true; if (animationRef.current) cancelAnimationFrame(animationRef.current); };
  }, [src, dimensions, objectFit, animated, animationSpeed, applyDithering]);

  return (
    <div ref={containerRef} className={cn("relative h-full w-full", className)}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" style={{ imageRendering: "pixelated" }} aria-label="Dithered image" role="img" />
    </div>
  );
};

export default DitherShader;
