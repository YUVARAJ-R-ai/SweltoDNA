# Svelto Splice Explorer: Frontend Design

**Status:** awaiting approval · **Closes:** #7, #8, #9 (see acceptance mapping) · **Date:** 2026-10-09

## Purpose

A demo-ready web app for the Sprint 2 review that lets someone zoom from a brain to a single base pair, change one letter, and see what happens to splicing. It is built from the approved prototype (`.superpowers/brainstorm/…/content/svelto-v5.html`). The prototype settles every visual and interaction decision, so this document covers only structure, contracts, and testing.

## Decisions

| Topic | Decision | Why |
| :--- | :--- | :--- |
| Framework | Next.js 15 (App Router), TypeScript, Tailwind v4, shadcn/ui | #7 says Next 14, but shadcn's current Tailwind v4 setup targets Next 15 / React 19. Same App Router model. |
| 3D | Plain `three` in one client component, driven by an imperative `SceneEngine` class | The prototype is already an imperative frame loop with springs. React Three Fiber would add a reconciler between React state and 60 fps updates for no gain. |
| Ribbon | `@tanstack/react-virtual` horizontal virtualizer | Required by #7. |
| State | `zustand` store | Selection, mutation stack (undo), scan results, and zoom level are shared between the canvas and the DOM chrome. |
| Dither | `components/ui/dither-shader.tsx` (vendored), with the two review fixes applied (stale `src`, memoized colors) | Used by the Snapshot sheet. |
| Location | `web/` at the repository root | `.gitignore` already anticipates `web/`. |

## Module layout (`web/`)

```
app/page.tsx                 server shell → <Explorer/> (client)
components/explorer.tsx      composes canvas + chrome, owns keyboard shortcuts
components/ui/               shadcn primitives + dither-shader.tsx
components/chrome/           toolbar (breadcrumbs), depth-scrubber, inspector, transcript-diagram,
                             sequence-ribbon, radial-switcher, performance-card, dynamic-island,
                             snapshot-sheet, about-popover
scene/engine.ts              SceneEngine: renderer, springs, levels, camera poses, capture()
scene/{brain,chromosome,helix,landscape,heat}.ts   scene builders (ported from the prototype)
lib/splice/sequence.ts       10 kb synthetic region (later: real MAPT region from #2)
lib/splice/scorer.ts         heuristic motif scorer, deltaAt, consequence, ISM scan (pure)
lib/splice/client.ts         SpliceClient interface + LocalHeuristicClient + WebSocketClient (#6)
lib/store.ts                 zustand store
lib/springs.ts               spring(), project(), rubber() (Apple-style damping + response)
```

Each unit has one job: the scene never reads the DOM, the chrome never touches `three`, and both talk only through the store and `SceneEngine`'s small API (`setZoomTarget`, `setWindow`, `select`, `animateMutation`, `startScanSweep`, `capture`).

## Data contract with #6 (`/ws/splice-session`)

The client sends #6's payload unchanged. The response extends #6 with what #8 needs to draw the tracks:

```jsonc
{
  "session_id": "uuid4", "status": "success",
  "window_start": 0, "window_end": 1024,
  "p_ref": [[0.98, 0.01, 0.01], ...],    // NEW: L×3 [neither, donor, acceptor], reference
  "p_mut": [[0.98, 0.01, 0.01], ...],    // NEW: L×3, mutant
  "delta_scores": { "donor_gain": [...], "donor_loss": [...], "acceptor_gain": [...], "acceptor_loss": [...] },
  "latency_ms": 42.5,
  "telemetry": {                          // optional; absent until #4 measures it
    "vanilla_latency_ms": 118.2, "speedup_ratio": 2.78, "acceptance_rate": 0.78,
    "active_flops_saved": 0.64, "source": "measured" | "simulated"
  }
}
```

`SpliceClient` hides the transport. `LocalHeuristicClient` (in-browser scorer) is the default and keeps the demo working with no server. `WebSocketClient` is used when `NEXT_PUBLIC_SPLICE_WS` is set. The HUD shows a "Simulated" label unless `telemetry.source === "measured"`.

## Honesty rules (carried over from the review)

- Splice scores from the local scorer are labelled "Heuristic". Scores from the server are labelled with the model name it reports.
- Performance figures say "Simulated" until #4 supplies measured values.
- The sequence and coordinates are labelled synthetic/illustrative until #2 provides the real MAPT region.

## Acceptance mapping

| Issue criterion | How it is met | Verified by |
| :--- | :--- | :--- |
| #7 60 fps over 10 kb, no DOM explosion | react-virtual renders only visible chips | Playwright: chip count < 200 at any scroll position |
| #7 radial switcher fires the edit within 16 ms | edit dispatched synchronously in the click handler | Unit test on handler timing |
| #7 pinned ruler tracks scroll | ruler is inside the virtualized row | Playwright scroll + screenshot |
| #7 keyboard ← → A C G T | explorer shortcuts | Playwright keyboard test |
| #8 reference vs mutant tracks aligned with the ribbon | landscape (3D) plus a 2D dual track under the ribbon sharing the virtualizer's scroll offset | Playwright alignment check |
| #8 hover tooltip: coordinate, class, Δ, tier | tooltip on track hover | Playwright hover test |
| #8 update < 30 ms without full re-render | landscape morph is a spring on a typed array | Performance mark in test |
| #9 HUD panel, live per-response updates, green above 2× | performance card bound to response telemetry | Unit test with mocked responses |
| #9 "accurate speedup" | **Only once #4 reports measured telemetry.** Until then it is labelled simulated, so #9 stays open until #4 lands. | n/a |

## Testing

- **Vitest:** scorer (known values: donor +2 T→A gives donor loss 1.00 and cryptic gain 0.62 at +16 bp; acceptor A→C gives loss 0.91), consequence ("exon 10 extended by 17 nt, frameshift"), springs, store undo/reset, both `SpliceClient`s against a fake socket.
- **Playwright** (Chromium, headless GPU fallback): load, zoom with keyboard to level 2, mutate with keys, assert the inspector says "High impact", open Snapshot, check the chip count bound, and run an axe accessibility scan.
- `prefers-reduced-motion`, `prefers-reduced-transparency`, and `prefers-contrast` are honored (from the prototype).

## Out of scope

Real model inference in the browser, authentication, and a light theme.
