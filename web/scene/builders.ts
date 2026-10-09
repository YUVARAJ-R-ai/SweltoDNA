/**
 * Scene builders, ported from the approved prototype. Each returns plain three.js objects plus the
 * handles the engine animates; none of them touch the DOM or the store.
 */
import * as THREE from "three";
import type { Base } from "@/lib/splice/sequence";
import { COMPLEMENT } from "@/lib/splice/sequence";

export const WIN = 64;            // bases on the helix (matches lib/store WINDOW)
export const RISE = 0.16;
export const RAD = 0.55;
export const HELIX_Y = 0.9;
export const LAND_Y = -1.9;
export const xAt = (k: number) => (k - WIN / 2) * RISE;

export const BASE_HEX: Record<Base, number> = { A: 0x10b981, C: 0x3b82f6, G: 0xf59e0b, T: 0xef4444 };
export const HOT = new THREE.Vector3(0.78, -0.2, 0.28);
/** The chromosome faces the direction the camera approaches it from, so its X shape reads. */
export const FACE = 1.18;
/** Scale of the sequence root inside the chromosome band. */
export const SEQ_SCALE = 0.02;
export const WORLD_PER_SEQ_UNIT = 0.34 * SEQ_SCALE;

const rand = Math.random;

/* ───────── level 0: brain point cloud + white-matter tractography ───────── */
export function buildBrain() {
  const group = new THREE.Group();
  const brainMat = new THREE.PointsMaterial({ size: 0.012, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  {
    const N = 30000, pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
    const cool = new THREE.Color(0x9fb4ff), warm = new THREE.Color(0xd9dcf0), c = new THREE.Color();
    for (let i = 0; i < N; i++) {
      let x: number, y: number, z: number;
      if (i < N * 0.94) {
        const v = new THREE.Vector3().randomDirection(); x = v.x; y = v.y; z = v.z;
        const fold = 0.07 * Math.sin(10 * x + 4 * y) * Math.cos(8 * z + 3 * y) + 0.045 * Math.sin(19 * y + 13 * z);
        const r = (0.9 + rand() * 0.1) * (1 + fold), hemi = x >= 0 ? 1 : -1;
        x = x * r * 0.8 + hemi * 0.06; y = y * r * 0.72; z = z * r * 1.08;
        if (y < -0.3) y = -0.3 + (y + 0.3) * 0.45;
      } else {
        const t = rand(), a = rand() * Math.PI * 2, rr = 0.11 * (1 - t * 0.4);
        x = Math.cos(a) * rr; y = -0.3 - t * 0.85; z = -0.25 - t * 0.15 + Math.sin(a) * rr;
      }
      pos.set([x, y, z], i * 3);
      c.copy(cool).lerp(warm, rand()).multiplyScalar(0.1 + rand() * 0.15);
      col.set([c.r, c.g, c.b], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    group.add(new THREE.Points(g, brainMat));
  }

  // Direction-encoded colour, DTI convention: red L–R, green A–P, blue S–I.
  const tractMat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uOpacity: { value: 1 } },
    vertexShader: `attribute vec3 dcol; attribute float u; attribute float seed; varying vec3 vC; varying float vU; varying float vS;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main(){ vC = dcol; vU = u; vS = seed; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `uniform float uTime, uOpacity; varying vec3 vC; varying float vU; varying float vS;
      #include <logdepthbuf_pars_fragment>
      void main(){
      #include <logdepthbuf_fragment>
        float p = fract(vU - uTime * 0.16 + vS);
        float pulse = smoothstep(0.0, 0.05, p) * (1.0 - smoothstep(0.05, 0.2, p));
        gl_FragColor = vec4(vC * (0.55 + pulse * 2.0), (0.045 + pulse * 0.5) * uOpacity);
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const T = 260, P = 32, nSeg = T * (P - 1);
  const pos = new Float32Array(nSeg * 6), col = new Float32Array(nSeg * 6), uu = new Float32Array(nSeg * 2), sd = new Float32Array(nSeg * 2);
  const shell = () => { const v = new THREE.Vector3().randomDirection(); const hemi = v.x >= 0 ? 1 : -1; return v.set(v.x * 0.66 + hemi * 0.05, Math.max(-0.24, v.y * 0.6), v.z * 0.92); };
  const side = (sx: number) => { let v: THREE.Vector3; do v = shell(); while (Math.sign(v.x) !== sx || v.y < -0.12); return v; };
  const Q = (a: THREE.Vector3, c: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3) =>
    out.set(0, 0, 0).addScaledVector(a, (1 - t) ** 2).addScaledVector(c, 2 * (1 - t) * t).addScaledVector(b, t * t);
  const q = new THREE.Vector3(), q2 = new THREE.Vector3(), tan = new THREE.Vector3();
  let o = 0;
  for (let t = 0; t < T; t++) {
    const kind = t / T, sx = rand() < 0.5 ? 1 : -1;
    let a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3;
    if (kind < 0.62) {                     // association fibres: U-shaped arcs within a hemisphere
      a = side(sx); b = side(sx); let tries = 0;
      while ((a.distanceTo(b) < 0.45 || a.distanceTo(b) > 1.1) && tries++ < 20) b = side(sx);
      c = a.clone().add(b).multiplyScalar(0.31); c.x = (a.x + b.x) * 0.38;
    } else if (kind < 0.86) {              // callosal fibres: arch across the midline
      a = side(1); b = side(-1); b.z = a.z * 0.8 + (rand() - 0.5) * 0.2;
      c = new THREE.Vector3(0, 0.3 + rand() * 0.08, ((a.z + b.z) / 2) * 0.8);
    } else {                               // projection fibres: cortex into the brainstem
      a = side(sx); a.y = Math.abs(a.y) + 0.2;
      b = new THREE.Vector3((rand() - 0.5) * 0.08, -0.62, -0.3 + (rand() - 0.5) * 0.06);
      c = new THREE.Vector3(a.x * 0.35, 0.02, (a.z - 0.25) * 0.5);
    }
    const seed = rand();
    for (let i = 0; i < P - 1; i++) {
      const s0 = i / (P - 1), s1 = (i + 1) / (P - 1);
      Q(a, c, b, s0, q); Q(a, c, b, s1, q2); tan.subVectors(q2, q).normalize();
      const cr = Math.abs(tan.x), cg = Math.abs(tan.z), cb = Math.abs(tan.y);
      pos.set([q.x, q.y, q.z, q2.x, q2.y, q2.z], o * 6); col.set([cr, cg, cb, cr, cg, cb], o * 6);
      uu.set([s0, s1], o * 2); sd.set([seed, seed], o * 2); o++;
    }
  }
  const tg = new THREE.BufferGeometry();
  tg.setAttribute("position", new THREE.BufferAttribute(pos, 3)); tg.setAttribute("dcol", new THREE.BufferAttribute(col, 3));
  tg.setAttribute("u", new THREE.BufferAttribute(uu, 1)); tg.setAttribute("seed", new THREE.BufferAttribute(sd, 1));
  const tracts = new THREE.LineSegments(tg, tractMat);
  group.add(tracts);

  const hotMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0x0a84ff).multiplyScalar(3), transparent: true });
  const hot = new THREE.Mesh(new THREE.SphereGeometry(0.035, 24, 24), hotMat);
  hot.position.copy(HOT); group.add(hot);
  return { group, brainMat, tractMat, tracts, hot, hotMat };
}

/* ───────── level 1: chromosome 17, living inside the brain at the hotspot ───────── */
export function buildChromosome(parent: THREE.Object3D) {
  const group = new THREE.Group();
  group.position.copy(HOT); group.scale.setScalar(0.34); group.rotation.set(0, FACE, 0.25);
  parent.add(group);
  const mat = new THREE.PointsMaterial({ size: 0.0055, vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
  const arms = [[-0.32, 0.75], [0.32, 0.75], [-0.38, -1.25], [0.38, -1.25]];
  const N = 14000, pos = new Float32Array(N * 3), col = new Float32Array(N * 3), c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const arm = arms[i % 4], t = Math.pow(rand(), 0.9);
    const bow = Math.sin(t * Math.PI) * 0.06 * Math.sign(arm[0]);
    const rad = 0.16 * (1 - 0.25 * t) * Math.sqrt(rand()), a = rand() * Math.PI * 2;
    pos.set([arm[0] * t + bow + Math.cos(a) * rad, arm[1] * t, Math.sin(a) * rad], i * 3);
    const hotBand = arm[1] < 0 && arm[0] > 0 && t > 0.41 && t < 0.49;    // 17q21.31
    if (hotBand) c.set(0x0a84ff).multiplyScalar(0.9);
    else c.set(Math.floor(t * 12) % 2 ? 0xc7c7cc : 0x48484a).multiplyScalar(0.18 + rand() * 0.12);
    col.set([c.r, c.g, c.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const points = new THREE.Points(g, mat);
  group.add(points);
  const band = new THREE.Group();
  band.position.set(0.38 * 0.45 + Math.sin(0.45 * Math.PI) * 0.06, -1.25 * 0.45, 0);
  group.add(band);
  return { group, mat, points, band };
}

/* ───────── levels 2–3: B-DNA helix (backbone tubes, stacked base plates) ───────── */
const AXIS = new THREE.Vector3(0, 1, 0);
export function plateMatrix(a: THREE.Vector3, b: THREE.Vector3, f: number, out: THREE.Matrix4) {
  const Y = b.clone().sub(a), L = Y.length(); Y.normalize();
  const X = new THREE.Vector3().crossVectors(Y, AXIS).normalize();
  const len = Math.max(0.02, (L - 0.03) * f), c = a.clone().addScaledVector(Y, len / 2 + 0.015);
  return out.makeBasis(X, Y, AXIS).scale(new THREE.Vector3(1, len, 1)).setPosition(c);
}

export function buildHelix(track: <M extends THREE.Material>(m: M) => M) {
  const outer = new THREE.Group(); outer.rotation.z = -Math.PI / 2; outer.position.y = HELIX_Y;
  const spin = new THREE.Group(); outer.add(spin);
  const pearl = track(new THREE.MeshPhysicalMaterial({ color: 0xe9ecf3, roughness: 0.2, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.1 }));
  const beads = new THREE.InstancedMesh(new THREE.SphereGeometry(0.058, 18, 18), pearl, WIN * 2);
  const rungs = new THREE.InstancedMesh(new THREE.BoxGeometry(0.15, 1, 0.045), track(new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0 })), WIN * 2);
  spin.add(beads, rungs);
  const halves: [THREE.Vector3, THREE.Vector3][] = [];
  const m = new THREE.Matrix4(), s1: THREE.Vector3[] = [], s2: THREE.Vector3[] = [];
  for (let k = 0; k < WIN; k++) {
    const a = k * ((Math.PI * 2) / 10.5), y = (k - WIN / 2) * RISE;
    const p1 = new THREE.Vector3(Math.cos(a) * RAD, y, Math.sin(a) * RAD), p2 = new THREE.Vector3(Math.cos(a + 2.4) * RAD, y, Math.sin(a + 2.4) * RAD);
    const mid = p1.clone().lerp(p2, 0.5);
    s1.push(p1); s2.push(p2); halves.push([p1, mid], [p2, mid]);
    beads.setMatrixAt(k * 2, m.makeTranslation(p1.x, p1.y, p1.z)); beads.setMatrixAt(k * 2 + 1, m.makeTranslation(p2.x, p2.y, p2.z));
    rungs.setMatrixAt(k * 2, plateMatrix(p1, mid, 1, m)); rungs.setMatrixAt(k * 2 + 1, plateMatrix(p2, mid, 1, m));
  }
  for (const pts of [s1, s2]) spin.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), WIN * 8, 0.034, 10, false), pearl));
  const tc = new THREE.Color();
  const paintPair = (k: number, b: Base) => {
    rungs.setColorAt(k * 2, tc.setHex(BASE_HEX[b]));
    rungs.setColorAt(k * 2 + 1, tc.setHex(BASE_HEX[COMPLEMENT[b]]).multiplyScalar(0.7));
  };
  const setPlate = (k: number, f: number) => {
    for (let j = 0; j < 2; j++) { const [a, b] = halves[k * 2 + j]; rungs.setMatrixAt(k * 2 + j, plateMatrix(a, b, f, m)); }
    rungs.instanceMatrix.needsUpdate = true;
  };
  return { outer, spin, rungs, paintPair, setPlate };
}

/* ───────── landscape: mutant surface + reference outline ───────── */
export const TW = WIN * RISE, TD = 3.2, SX = 256, SZ = 32;
export function buildLandscape(track: <M extends THREE.Material>(m: M) => M) {
  const g = new THREE.PlaneGeometry(TW, TD, SX, SZ); g.rotateX(-Math.PI / 2); g.translate(-RISE / 2, 0, 0);
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 3), 3));
  const surf = new THREE.Mesh(g, track(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0, side: THREE.DoubleSide, opacity: 0.95 })));
  surf.position.y = LAND_Y;
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array((SX + 1) * 3), 3));
  // #8: reference track in blue
  const refLine = new THREE.Line(lg, track(new THREE.LineBasicMaterial({ color: new THREE.Color(0x64d2ff), opacity: 0.95 })));
  refLine.position.y = LAND_Y + 0.015;
  const zFall = Float32Array.from({ length: SZ + 1 }, (_, r) => { const z = -TD / 2 + (r * TD) / SZ; return Math.exp(-(z * z) / 0.9); });
  const colX = Float32Array.from({ length: SX + 1 }, (_, c) => -TW / 2 - RISE / 2 + (c * TW) / SX);
  const cLow = new THREE.Color(0x2c2c2e), cHigh = new THREE.Color(0xd1d1d6), cGain = new THREE.Color(0xff9f0a), cLoss = new THREE.Color(0xff453a), cv = new THREE.Color();
  const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
  function write(profRef: Float32Array, profMut: Float32Array) {
    const pa = surf.geometry.attributes.position as THREE.BufferAttribute, ca = surf.geometry.attributes.color as THREE.BufferAttribute;
    for (let i = 0; i < pa.count; i++) {
      const c = i % (SX + 1), r = Math.floor(i / (SX + 1)), h = profMut[c] * zFall[r];
      pa.setY(i, h);
      cv.copy(cLow).lerp(cHigh, clamp(h / 1.8, 0, 1));
      const d = profMut[c] - profRef[c];
      if (d > 0.03) cv.lerp(cGain, clamp(d * 1.4, 0, 0.9)); else if (d < -0.03) cv.lerp(cLoss, clamp(-d * 1.4, 0, 0.9) * zFall[r]);
      ca.setXYZ(i, cv.r, cv.g, cv.b);
    }
    pa.needsUpdate = true; ca.needsUpdate = true; surf.geometry.computeVertexNormals();
    const la = refLine.geometry.attributes.position as THREE.BufferAttribute;
    for (let c = 0; c <= SX; c++) la.setXYZ(c, colX[c], profRef[c], 0);
    la.needsUpdate = true;
  }
  /** Gaussian-smoothed profile of max(donor, acceptor) across the helix window. */
  function profile(donor: (k: number) => number, acceptor: (k: number) => number, out: Float32Array) {
    for (let c = 0; c <= SX; c++) {
      const x = colX[c], kc = Math.round(x / RISE + WIN / 2); let v = 0;
      for (let k = kc - 3; k <= kc + 3; k++) {
        if (k < 0 || k >= WIN) continue;
        const p = Math.max(donor(k), acceptor(k)); if (!p) continue;
        const dx = x - xAt(k); v += p * Math.exp(-(dx * dx) / (2 * 0.1 * 0.1));
      }
      out[c] = v * 2.0;
    }
  }
  return { surf, refLine, write, profile, zFall, colX };
}
