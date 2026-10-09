/**
 * SceneEngine: one continuous zoom (brain → chromosome → helix → landscape) driven by springs.
 * Reads application state from the store; reports per-frame values to the DOM chrome via onFrame.
 */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import type { ExplorerStore, ExplorerState } from "@/lib/store";
import { WINDOW } from "@/lib/store";
import type { Base } from "@/lib/splice/sequence";
import { project, rubberband, stepSpring, type SpringState } from "@/lib/springs";
import {
  buildBrain, buildChromosome, buildHelix, buildLandscape, FACE, HELIX_Y, LAND_Y, RISE, SEQ_SCALE, SX, SZ, TD, TW, WIN, WORLD_PER_SEQ_UNIT, xAt,
} from "./builders";

export const LEVELS = ["Brain", "Chromosome 17", "MAPT exon 10", "Splice landscape"] as const;

export interface ScreenAnchor { x: number; y: number; visible: boolean }
export interface FrameInfo {
  z: number;
  level: number;
  fades: { brain: number; chromo: number; seq: number };
  anchors: { hot: ScreenAnchor; band: ScreenAnchor; loss: ScreenAnchor; gain: ScreenAnchor };
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const smoothstep = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

interface Keyframe { tgt: THREE.Vector3; dist: number; ph: number; th: number }

export class SceneEngine {
  readonly zoom: SpringState = { x: 0, v: 0 };
  zoomTarget = 0;
  onFrame: ((f: FrameInfo) => void) | null = null;
  onRungTap: ((i: number, clientX: number, clientY: number) => void) | null = null;
  onInteract: (() => void) | null = null;

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private brain: ReturnType<typeof buildBrain>;
  private chromo: ReturnType<typeof buildChromosome>;
  private seqRoot = new THREE.Group();
  private seqMats: THREE.Material[] = [];
  private helix: ReturnType<typeof buildHelix>;
  private land: ReturnType<typeof buildLandscape>;
  private selRing: THREE.Mesh; private wave: THREE.Mesh; private waveT = 9;
  private mutRings: THREE.Mesh[];
  private heat: THREE.InstancedMesh;
  private sweep = new THREE.Group(); private sweepMat: THREE.MeshBasicMaterial; private sweepFill: THREE.MeshBasicMaterial;
  private lossLine: THREE.Mesh; private gainLine: THREE.Mesh;
  private flowGeo: THREE.BufferGeometry; private flowSpeed: Float32Array;
  private profRef = new Float32Array(SX + 1); private profMut = new Float32Array(SX + 1);
  private profRefT = new Float32Array(SX + 1); private profMutT = new Float32Array(SX + 1);
  private KF: Keyframe[];
  private orbit = { th: 0, ph: 0, vth: 0, vph: 0 };
  private drag: { x: number; y: number; moved: boolean; hist: [number, number, number][] } | null = null;
  private scrubbing = false;
  private wheelHist: [number, number][] = []; private wheelTimer = 0;
  private unzip: { k: number; from: Base; t: number; swapped: boolean } | null = null;
  private scan: { positions: number[]; k: number; acc: number; step: (i: number) => void; done: () => void } | null = null;
  private state: ExplorerState;
  private unsub: () => void;
  private raf = 0; private last = performance.now(); private t = 0;
  private interacted = false;
  private ray = new THREE.Raycaster(); private ndc = new THREE.Vector2();
  private tmp = new THREE.Vector3();
  private readonly reduced: boolean;

  constructor(private host: HTMLElement, private store: ExplorerStore) {
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.state = store.getState();
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, powerPreference: "high-performance", preserveDrawingBuffer: false }));
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.setSize(host.clientWidth, host.clientHeight);
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.domElement.style.touchAction = "none";
    host.appendChild(r.domElement);
    this.scene.background = new THREE.Color(0x000000);
    this.scene.environment = new THREE.PMREMGenerator(r).fromScene(new RoomEnvironment(), 0.04).texture;
    this.camera = new THREE.PerspectiveCamera(42, host.clientWidth / host.clientHeight, 0.001, 500);
    this.scene.add(new THREE.HemisphereLight(0xe6ecff, 0x0a0a0f, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(-3, 5, 4); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x9ab8ff, 0.8); rim.position.set(4, 1, -5); this.scene.add(rim);
    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(host.clientWidth, host.clientHeight), 0.42, 0.45, 0.82));
    this.composer.addPass(new OutputPass());

    this.addStars();
    this.brain = buildBrain(); this.scene.add(this.brain.group);
    this.chromo = buildChromosome(this.brain.group);
    this.seqRoot.rotation.z = -0.25; this.seqRoot.scale.setScalar(SEQ_SCALE); this.chromo.band.add(this.seqRoot);
    const track = <M extends THREE.Material>(m: M): M => { m.transparent = true; m.userData.base = m.opacity; this.seqMats.push(m); return m; };
    this.helix = buildHelix(track); this.seqRoot.add(this.helix.outer);
    this.land = buildLandscape(track); this.seqRoot.add(this.land.surf, this.land.refLine);

    const ring = (r0: number, t0: number, color: THREE.ColorRepresentation, k: number) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(r0, t0, 12, 96), track(new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(k) })));
      m.rotation.y = Math.PI / 2; this.seqRoot.add(m); return m;
    };
    this.selRing = ring(0.74, 0.018, 0x0a84ff, 2.6);
    this.mutRings = Array.from({ length: 16 }, () => { const m = ring(0.66, 0.012, 0xff9f0a, 1.6); m.visible = false; return m; });
    this.wave = new THREE.Mesh(new THREE.TorusGeometry(0.78, 0.014, 10, 96), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
    this.wave.rotation.y = Math.PI / 2; this.seqRoot.add(this.wave);

    this.heat = new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 16, 16), track(new THREE.MeshBasicMaterial({ color: 0xffffff })), WIN);
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let k = 0; k < WIN; k++) { this.heat.setMatrixAt(k, zero); this.heat.setColorAt(k, new THREE.Color(0, 0, 0)); }
    this.seqRoot.add(this.heat);
    this.sweepMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xffffff).multiplyScalar(1.6), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
    this.sweepFill = new THREE.MeshBasicMaterial({ color: 0x9ccaff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
    this.sweep.add(new THREE.Mesh(new THREE.TorusGeometry(0.86, 0.012, 10, 96), this.sweepMat), new THREE.Mesh(new THREE.CircleGeometry(0.86, 64), this.sweepFill));
    this.sweep.rotation.y = Math.PI / 2; this.sweep.position.y = HELIX_Y; this.seqRoot.add(this.sweep);

    const hair = (hex: number) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 1, 8), track(new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(2.2), opacity: 0 }))); this.seqRoot.add(m); return m; };
    this.lossLine = hair(0xff453a); this.gainLine = hair(0xff9f0a);

    const FLOW = 600, fp = new Float32Array(FLOW * 3); this.flowSpeed = new Float32Array(FLOW);
    for (let i = 0; i < FLOW; i++) { fp.set([-TW / 2 + Math.random() * TW, LAND_Y, (Math.random() - 0.5) * 1.6], i * 3); this.flowSpeed[i] = 0.4 + Math.random() * 0.8; }
    this.flowGeo = new THREE.BufferGeometry(); this.flowGeo.setAttribute("position", new THREE.BufferAttribute(fp, 3));
    this.seqRoot.add(new THREE.Points(this.flowGeo, track(new THREE.PointsMaterial({ color: 0xffffff, size: 0.035 * WORLD_PER_SEQ_UNIT, opacity: 0.35, depthWrite: false }))));

    this.KF = this.keyframes();
    this.paintHelix(); this.retarget(); this.profRef.set(this.profRefT); this.profMut.set(this.profMutT); this.land.write(this.profRef, this.profMut);
    this.unsub = store.subscribe((s, prev) => this.onState(s, prev));
    this.bindInput();
    addEventListener("resize", this.onResize);
    this.raf = requestAnimationFrame(this.frame);
  }

  /* ───────── public API ───────── */
  setZoomTarget(z: number) { this.zoomTarget = clamp(z, 0, 3); this.markInteracted(); }
  /** Depth scrubber: follow 1:1 while dragging, then hand the release velocity to the spring. */
  scrubTo(z: number) { this.scrubbing = true; let t = z; if (t < 0) t = rubberband(t, 1); if (t > 3) t = 3 + rubberband(t - 3, 1); this.zoomTarget = t; this.zoom.x = lerp(this.zoom.x, t, 0.6); this.markInteracted(); }
  scrubEnd(velocityPerSecond: number) { this.scrubbing = false; this.zoom.v = velocityPerSecond; this.zoomTarget = clamp(Math.round(this.zoomTarget + project(velocityPerSecond * 120) / 120), 0, 3); }
  resetOrbit() { Object.assign(this.orbit, { th: 0, ph: 0, vth: 0, vph: 0 }); }
  /** Runs fn(position) for each position at ~40/s, sweeping a ring of light along the helix. */
  runScan(positions: number[], step: (i: number) => void, done: () => void) { this.scan = { positions, k: 0, acc: 0, step, done }; }
  get scanning() { return !!this.scan; }
  /** Same task as the draw, so the drawing buffer is intact (no preserveDrawingBuffer needed). */
  capture(): string { this.composer.render(); return this.renderer.domElement.toDataURL("image/png"); }
  dispose() {
    cancelAnimationFrame(this.raf); this.unsub(); removeEventListener("resize", this.onResize);
    this.renderer.domElement.remove(); this.renderer.dispose();
  }

  /* ───────── store → scene ───────── */
  private onState(s: ExplorerState, prev: ExplorerState) {
    this.state = s;
    if (s.windowStart !== prev.windowStart) this.finishUnzip();
    if (s.seq !== prev.seq || s.windowStart !== prev.windowStart) this.paintHelix();
    if (s.tracks !== prev.tracks || s.windowStart !== prev.windowStart) this.retarget();
    const added = s.edits.length > prev.edits.length ? s.edits[s.edits.length - 1] : null;
    if (added && !this.reduced && added.position >= s.windowStart && added.position < s.windowStart + WINDOW) {
      this.finishUnzip();
      this.unzip = { k: added.position - s.windowStart, from: added.from, t: 0, swapped: false };
      this.wave.position.set(xAt(added.position - s.windowStart), HELIX_Y, 0); this.waveT = 0;
    }
  }
  private paintHelix() {
    const { seq, windowStart } = this.state;
    for (let k = 0; k < WIN; k++) this.helix.paintPair(k, seq[windowStart + k]);
    this.helix.rungs.instanceColor!.needsUpdate = true;
  }
  private finishUnzip() { if (!this.unzip) return; this.helix.setPlate(this.unzip.k, 1); this.unzip = null; this.paintHelix(); }
  private retarget() {
    const { tracks: t, windowStart: w0 } = this.state;
    const at = (arr: Float32Array | undefined, k: number) => { if (!t || !arr) return 0; const i = w0 + k - t.start; return i >= 0 && i < arr.length ? arr[i] : 0; };
    this.land.profile((k) => at(t?.refDonor, k), (k) => at(t?.refAcceptor, k), this.profRefT);
    this.land.profile((k) => at(t?.mutDonor, k), (k) => at(t?.mutAcceptor, k), this.profMutT);
  }

  /* ───────── camera ───────── */
  private keyframes(): Keyframe[] {
    this.scene.updateMatrixWorld(true);
    const w = (o: THREE.Object3D, x: number, y: number, z: number) => o.localToWorld(new THREE.Vector3(x, y, z));
    const S = WORLD_PER_SEQ_UNIT;
    return [
      { tgt: new THREE.Vector3(0, -0.1, 0), dist: 4.3, ph: 0.12, th: 0.62 },
      { tgt: w(this.chromo.group, 0.02, -0.22, 0), dist: 1.15, ph: 0.06, th: FACE },
      { tgt: w(this.seqRoot, 0, HELIX_Y * 0.25, 0), dist: 11.5 * S, ph: 0.06, th: FACE },
      { tgt: w(this.seqRoot, 0.3, LAND_Y + 0.2, 0), dist: 10.5 * S, ph: 0.42, th: FACE - 0.38 },
    ];
  }
  private pose(z: number) {
    const zc = clamp(z, 0, 3), i = Math.min(2, Math.floor(zc)), u = smoothstep(0, 1, zc - i), a = this.KF[i], b = this.KF[i + 1];
    const dist = Math.exp(lerp(Math.log(a.dist), Math.log(b.dist), u)) * Math.exp(-(z - zc) * 0.5);
    return { tgt: a.tgt.clone().lerp(b.tgt, u), dist, ph: lerp(a.ph, b.ph, u), th: lerp(a.th, b.th, u) };
  }

  /* ───────── input ───────── */
  private markInteracted() { if (!this.interacted) { this.interacted = true; this.onInteract?.(); } }
  private bindInput() {
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      this.drag = { x: e.clientX, y: e.clientY, moved: false, hist: [[e.clientX, e.clientY, performance.now()]] };
      this.orbit.vth = this.orbit.vph = 0;
    });
    el.addEventListener("pointermove", (e) => {
      const d = this.drag; if (!d) return;
      if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 6) return;      // hysteresis before committing to a drag
      if (!d.moved) { d.moved = true; d.x = e.clientX; d.y = e.clientY; this.markInteracted(); return; }
      this.orbit.th -= (e.clientX - d.x) * 0.0055;
      this.orbit.ph = clamp(this.orbit.ph + (e.clientY - d.y) * 0.0045, -1.1, 1.1);
      d.x = e.clientX; d.y = e.clientY;
      d.hist.push([e.clientX, e.clientY, performance.now()]); if (d.hist.length > 6) d.hist.shift();
    });
    el.addEventListener("pointerup", (e) => {
      const d = this.drag; this.drag = null; if (!d) return;
      if (d.moved) {                                                                   // inertia: hand off release velocity
        const [x0, y0, t0] = d.hist[0], [x1, y1, t1] = d.hist[d.hist.length - 1], dt = Math.max(16, t1 - t0);
        this.orbit.vth = (-(x1 - x0) / dt) * 0.0055 * 1000; this.orbit.vph = ((y1 - y0) / dt) * 0.0045 * 1000;
        return;
      }
      this.tap(e.clientX, e.clientY);
    });
    addEventListener("wheel", this.onWheel, { passive: false });
  }
  private onWheel = (e: WheelEvent) => {
    if ((e.target as HTMLElement).closest("[data-scroll-own]")) return;
    e.preventDefault(); this.markInteracted();
    const k = e.ctrlKey ? 0.012 : e.deltaMode === 1 ? 0.08 : 0.0028;
    let t = this.zoomTarget + e.deltaY * k;
    if (t < 0) t = rubberband(t, 1); if (t > 3) t = 3 + rubberband(t - 3, 1);
    this.zoomTarget = t;
    const now = performance.now(); this.wheelHist.push([t, now]); this.wheelHist = this.wheelHist.filter((h) => now - h[1] < 120);
    clearTimeout(this.wheelTimer);
    this.wheelTimer = window.setTimeout(() => {           // momentum-projected snap to the nearest level
      const h = this.wheelHist, v = h.length > 1 ? ((h[h.length - 1][0] - h[0][0]) / Math.max(16, h[h.length - 1][1] - h[0][1])) * 1000 : 0;
      this.zoomTarget = clamp(Math.round(this.zoomTarget + project(v * 120) / 120), 0, 3); this.wheelHist = [];
    }, 110);
  };
  private tap(x: number, y: number) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.ndc.set(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1);
    this.ray.setFromCamera(this.ndc, this.camera);
    const lvl = Math.round(this.zoom.x);
    if (lvl === 0) { if (this.ray.intersectObject(this.brain.hot)[0]) this.setZoomTarget(1); return; }
    if (lvl >= 2) {
      const hit = this.ray.intersectObject(this.helix.rungs)[0];
      if (hit?.instanceId !== undefined) this.onRungTap?.(this.state.windowStart + Math.floor(hit.instanceId / 2), x, y);
    }
  }
  private onResize = () => {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    this.renderer.setSize(w, h); this.composer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  };

  private addStars() {
    const N = 1400, p = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) { const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 60); p.set([v.x, v.y, v.z], i * 3); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    this.scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8e8e93, size: 1.4, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false })));
  }

  private screen(world: THREE.Vector3, show: boolean): ScreenAnchor {
    const p = this.tmp.copy(world).project(this.camera);
    const visible = show && p.z < 1 && Math.abs(p.x) < 1.05 && Math.abs(p.y) < 1.05;
    return { x: ((p.x + 1) / 2) * this.host.clientWidth, y: ((1 - p.y) / 2) * this.host.clientHeight, visible };
  }

  /* ───────── frame ───────── */
  private frame = (now: number) => {
    const dt = Math.min((now - this.last) / 1000, 1 / 30); this.last = now; this.t += dt;
    const { orbit, zoom } = this, t = this.t, s = this.state;

    if (!this.scrubbing) stepSpring(zoom, this.zoomTarget, dt, this.reduced ? 0.18 : 0.62, 1);
    if (!this.drag) {
      orbit.th += orbit.vth * dt; orbit.ph = clamp(orbit.ph + orbit.vph * dt, -1.1, 1.1);
      const decay = Math.pow(0.996, dt * 1000); orbit.vth *= decay; orbit.vph *= decay;
      if (Math.abs(zoom.v) > 0.05) { orbit.th *= 1 - Math.min(1, dt * 2.2); orbit.ph *= 1 - Math.min(1, dt * 2.2); }
      if (!this.reduced && !this.interacted && Math.round(zoom.x) === 0) orbit.th += dt * 0.05;   // gentle idle drift until first touch
    }
    const z = zoom.x, P = this.pose(z), th = P.th + orbit.th, ph = clamp(P.ph + orbit.ph, -1.25, 1.25);
    this.camera.position.set(P.tgt.x + P.dist * Math.sin(th) * Math.cos(ph), P.tgt.y + P.dist * Math.sin(ph), P.tgt.z + P.dist * Math.cos(th) * Math.cos(ph));
    this.camera.lookAt(P.tgt); this.camera.near = P.dist * 0.01; this.camera.updateProjectionMatrix();

    const fBrain = 1 - smoothstep(0.5, 1.1, z), fChromo = smoothstep(0.35, 0.9, z) * (1 - smoothstep(1.55, 1.95, z)), fSeq = smoothstep(1.5, 1.95, z);
    const b = this.brain;
    b.brainMat.opacity = fBrain; b.group.children[0].visible = fBrain > 0.01;
    b.tractMat.uniforms.uTime.value = this.reduced ? 0 : t; b.tractMat.uniforms.uOpacity.value = fBrain; b.tracts.visible = fBrain > 0.01;
    b.hotMat.opacity = fBrain; b.hot.visible = fBrain > 0.35; b.hot.scale.setScalar(1 + (this.reduced ? 0 : 0.12 * Math.sin(t * 2 * Math.PI)));
    this.chromo.mat.opacity = fChromo; this.chromo.points.visible = fChromo > 0.01;
    for (const m of this.seqMats) m.opacity = m.userData.base * fSeq;
    this.seqRoot.visible = fSeq > 0.01;

    const anchors = { hot: this.screen(b.hot.getWorldPosition(new THREE.Vector3()), fBrain > 0.6), band: this.screen(this.chromo.band.getWorldPosition(new THREE.Vector3()), fChromo > 0.7 && z < 1.4), loss: { x: 0, y: 0, visible: false }, gain: { x: 0, y: 0, visible: false } };
    if (this.seqRoot.visible) Object.assign(anchors, this.updateSequence(dt, t, fSeq, s));
    this.onFrame?.({ z, level: clamp(Math.round(z), 0, 3), fades: { brain: fBrain, chromo: fChromo, seq: fSeq }, anchors });

    this.composer.render();
    this.raf = requestAnimationFrame(this.frame);
  };

  private updateSequence(dt: number, t: number, fSeq: number, s: ExplorerState) {
    const w0 = s.windowStart;
    if (!this.reduced) this.helix.spin.rotation.y += dt * 0.22;

    // landscape morph toward the latest tracks
    const kk = 1 - Math.exp(-dt * 9); let moving = false;
    for (let c = 0; c <= SX; c++) {
      const a = this.profRefT[c] - this.profRef[c], b = this.profMutT[c] - this.profMut[c];
      if (Math.abs(a) > 1e-4 || Math.abs(b) > 1e-4) { moving = true; this.profRef[c] += a * kk; this.profMut[c] += b * kk; }
    }
    if (moving) this.land.write(this.profRef, this.profMut);

    // selection + edited positions
    const inWin = s.selected >= w0 && s.selected < w0 + WIN;
    this.selRing.visible = inWin; if (inWin) this.selRing.position.set(xAt(s.selected - w0), HELIX_Y, 0);
    let n = 0;
    for (let k = 0; k < WIN && n < this.mutRings.length; k++) if (s.seq[w0 + k] !== s.region.seq[w0 + k]) { this.mutRings[n].visible = true; this.mutRings[n++].position.set(xAt(k), HELIX_Y, 0); }
    for (; n < this.mutRings.length; n++) this.mutRings[n].visible = false;

    // unzip: the pair retracts to the backbone and the new base springs back in
    if (this.unzip) {
      const u = this.unzip; u.t += dt / 0.62; const tt = Math.min(1, u.t);
      const f = tt < 0.4 ? 1 - (tt / 0.4) * 0.92 : 1 - 0.92 * Math.exp(-(tt - 0.4) * 13) * Math.cos((tt - 0.4) * 17);
      this.helix.setPlate(u.k, f);
      if (tt < 0.4) { this.helix.paintPair(u.k, u.from); this.helix.rungs.instanceColor!.needsUpdate = true; }
      else if (!u.swapped) { u.swapped = true; this.paintHelix(); }
      if (tt >= 1) this.finishUnzip();
    }
    if (this.waveT < 0.7) { this.waveT += dt; this.wave.scale.setScalar(1 + this.waveT * 3.2); (this.wave.material as THREE.MeshBasicMaterial).opacity = (1 - this.waveT / 0.7) * 0.8 * fSeq; }
    else (this.wave.material as THREE.MeshBasicMaterial).opacity = 0;

    // saturation scan sweep
    if (this.scan) {
      const sc = this.scan; sc.acc += dt * (this.reduced ? 600 : 40);
      while (sc.acc >= 1 && sc.k < sc.positions.length) { sc.acc -= 1; sc.step(sc.positions[sc.k]); sc.k++; }
      this.sweep.position.x = xAt(clamp(sc.positions[Math.min(sc.k, sc.positions.length - 1)] - w0, 0, WIN - 1));
      this.sweepMat.opacity = 0.9 * fSeq; this.sweepFill.opacity = 0.07 * fSeq;
      if (sc.k >= sc.positions.length) { this.scan = null; sc.done(); }
    } else { this.sweepMat.opacity *= 0.85; this.sweepFill.opacity *= 0.85; }
    const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), Pv = new THREE.Vector3(), Sv = new THREE.Vector3(), C = new THREE.Color();
    for (let k = 0; k < WIN; k++) {
      const h = s.scan[w0 + k], sc = h < 0.05 ? 0 : 0.3 + h * 0.85;
      this.heat.setMatrixAt(k, M.compose(Pv.set(xAt(k), HELIX_Y, 0), Q, Sv.setScalar(sc)));
      this.heat.setColorAt(k, heatColor(Math.max(0, h), C).multiplyScalar(1 + Math.max(0, h) * 2.4));
    }
    this.heat.instanceMatrix.needsUpdate = true; this.heat.instanceColor!.needsUpdate = true;

    // Δ hairlines from the landscape up to the helix
    const d = s.delta, out = { loss: { x: 0, y: 0, visible: false }, gain: { x: 0, y: 0, visible: false } };
    const marks: [THREE.Mesh, "loss" | "gain", number][] = [
      [this.lossLine, "loss", d && Math.max(d.donorLoss, d.acceptorLoss) > 0.2 ? (d.donorLoss >= d.acceptorLoss ? d.donorLossAt : d.acceptorLossAt) : -1],
      [this.gainLine, "gain", d && Math.max(d.donorGain, d.acceptorGain) > 0.2 ? (d.donorGain >= d.acceptorGain ? d.donorGainAt : d.acceptorGainAt) : -1],
    ];
    for (const [line, key, idx] of marks) {
      const on = idx >= w0 && idx < w0 + WIN, mat = line.material as THREE.MeshBasicMaterial;
      mat.opacity += ((on ? 0.9 : 0) * fSeq - mat.opacity) * kk;
      if (!on) continue;
      const k = idx - w0, c = clamp(Math.round(((xAt(k) + TW / 2 + RISE / 2) / TW) * SX), 0, SX);
      const bottom = LAND_Y + Math.max(this.profMut[c], this.profRef[c]), top = HELIX_Y - 0.62;
      line.position.set(xAt(k), (bottom + top) / 2, 0); line.scale.set(1, Math.max(0.01, top - bottom), 1);
      out[key] = this.screen(this.seqRoot.localToWorld(new THREE.Vector3(xAt(k), (bottom + top) / 2, 0)), fSeq > 0.8);
    }

    // transcript flow along the mutant surface (5′ → 3′)
    if (!this.reduced) {
      const a = this.flowGeo.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < a.count; i++) {
        let x = a.getX(i) + this.flowSpeed[i] * dt; if (x > TW / 2) x -= TW;
        const c = clamp(Math.round(((x + TW / 2 + RISE / 2) / TW) * SX), 0, SX), zr = clamp(Math.round(((a.getZ(i) + TD / 2) / TD) * SZ), 0, SZ);
        a.setX(i, x); a.setY(i, LAND_Y + this.profMut[c] * this.land.zFall[zr] + 0.03);
      }
      a.needsUpdate = true;
    }
    void t;
    return out;
  }
}

const hBlue = new THREE.Color(0x0a84ff), hOr = new THREE.Color(0xff9f0a), hRed = new THREE.Color(0xff453a);
export function heatColor(h: number, out: THREE.Color) { return h < 0.5 ? out.copy(hBlue).lerp(hOr, h / 0.5) : out.copy(hOr).lerp(hRed, (h - 0.5) / 0.5); }
export function heatCss(h: number) { return `#${heatColor(h, new THREE.Color()).getHexString()}`; }
