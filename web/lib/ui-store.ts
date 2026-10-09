import { createStore } from "zustand/vanilla";

export interface Radial { position: number; x: number; y: number }
export interface UIState {
  perfOpen: boolean; aboutOpen: boolean;
  snapshot: string | null;        // captured frame (data URL) while the sheet is open
  tourOn: boolean; soundOn: boolean; hintGone: boolean; scanning: boolean;
  radial: Radial | null;
  set(p: Partial<Omit<UIState, "set">>): void;
}
export const createUIStore = () =>
  createStore<UIState>()((set) => ({
    perfOpen: false, aboutOpen: false, snapshot: null, tourOn: false, soundOn: true, hintGone: false, scanning: false, radial: null,
    set: (p) => set(p),
  }));
export type UIStore = ReturnType<typeof createUIStore>;
