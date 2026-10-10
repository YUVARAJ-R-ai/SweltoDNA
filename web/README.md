# Svelto Splice Explorer (`web/`)

Zoom from a brain to a single base pair, change one letter, and see how splicing responds. Closes the frontend issues #7 (virtualized ribbon + radial switcher), #8 (reference vs mutant tracks), and #9 (telemetry HUD).

Design: [`docs/superpowers/specs/2026-10-09-splice-explorer-frontend-design.md`](../docs/superpowers/specs/2026-10-09-splice-explorer-frontend-design.md)

## Run

```bash
npm install
npm run dev          # http://localhost:3000
npm run build && npm start
```

By default scores come from the in-browser heuristic scorer, so no server is needed. To use the #6 WebSocket server instead:

```bash
NEXT_PUBLIC_SPLICE_WS=ws://localhost:8000/ws/splice-session npm run dev
```

The UI labels every number by where it came from: splice scores show the engine name ("Heuristic scorer" or the model the server reports), the performance card says "Simulated" until the server sends `telemetry.source: "measured"`, and the sequence and coordinates are marked synthetic/illustrative.

## Use

| Input | Action |
| :--- | :--- |
| Scroll / pinch, `1`–`4`, breadcrumbs, depth slider | Zoom brain → chromosome 17 → MAPT exon 10 → splice landscape |
| Drag | Look around (with inertia) |
| Click a helix rung or ribbon chip | Radial base switcher |
| `←` `→` (`Shift` ×10), `A` `C` `G` `T` | Move and rewrite the selected base |
| `⌘Z` / `Ctrl+Z` | Undo |
| `P` | Guided tour (any input stops it) |
| `S` | Snapshot: 1-bit dithered print of the view, saved as PNG |

Presenter shortcuts: append `#z=2` (jump to the helix) or `#z=2&mut` (jump and apply the demo edit) to the URL.

## Test

```bash
npm test                                         # Vitest: scorer, springs, client (#6 contract), store
CHROMIUM_PATH=/usr/bin/chromium npm run test:e2e # Playwright: #7/#8/#9 acceptance flows + axe scan
```

## Layout

```
app/                     Next.js App Router shell
components/explorer.tsx  composition root: store, engine, shortcuts, tour
components/chrome/       toolbar, depth scrubber, inspector, ribbon + dual track, radial, HUD, island, snapshot
components/ui/           shadcn primitives + dither-shader (vendored, two fixes)
scene/                   three.js SceneEngine + scene builders (no DOM, no React)
lib/splice/              synthetic region, heuristic scorer, SpliceClient (local + WebSocket)
lib/                     store (zustand), springs, sound, tour
```
