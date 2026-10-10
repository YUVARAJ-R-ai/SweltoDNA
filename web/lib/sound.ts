/** Sound fires on the same frame as its visual, only for meaningful moments (select, edit, scan done, save). */
export class Sound {
  enabled = true;
  private ctx: AudioContext | null = null;

  /** Browsers allow audio only after a user gesture; call from a pointerdown handler. */
  unlock() {
    if (!this.ctx) this.ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    else if (this.ctx.state === "suspended") void this.ctx.resume();
  }

  tone(freq: number, dur: number, type: OscillatorType = "sine", gain = 0.05, when = 0) {
    const a = this.ctx;
    if (!this.enabled || !a) return;
    const t = a.currentTime + when, o = a.createOscillator(), g = a.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur + 0.03);
  }

  select() { this.tone(1320, 0.04, "sine", 0.02); }
  edit(max: number) { this.tone(660, 0.09, "triangle", 0.045); this.tone(990, 0.16, "sine", 0.035, 0.05); if (max > 0.8) this.tone(110, 0.5, "sine", 0.12); }
  scanTick(v: number) { this.tone(360 + v * 880, 0.035, "sine", 0.012); }
  chord() { this.tone(523, 0.18, "sine", 0.035); this.tone(659, 0.18, "sine", 0.03, 0.08); this.tone(784, 0.3, "sine", 0.03, 0.16); }
  shutter() { this.tone(2200, 0.03, "square", 0.012); this.tone(1300, 0.06, "sine", 0.03, 0.03); }
}
