/**
 * DebugScene — Renders a single self-attention block in isolation.
 *
 * Structure:
 *                [Kᵀ]          <- transposed K (4×8), above dot product space
 *   [Q]        [Attn]         <- Q (8×4) left, Attn (8×8) center
 *               [Vᵀ]          <- transposed V (4×8), rotated 90° toward viewer at Attn top edge
 *             [Output]         <- output projection box
 *
 * Q and Kᵀ are positioned so that the empty space between them is where
 * the 8×8 dot product result (Q × Kᵀ) would appear.
 * Vᵀ is rotated 90° toward the viewer around Attn's top edge, with cube-shaped
 * cells. Its 8 columns align with Attn's columns, showing how attention scores
 * weight V's sequence positions.
 *
 * Each TensorGrid cell is colored by a heatmap (indigo->blue->teal->amber->red)
 * driven by simulated weight magnitudes that evolve smoothly over time.
 */

import * as THREE from "three";
import { TensorGrid } from "./TensorGrid";

/** Canvas-based text sprite for component labels. */
function makeLabel(text: string, size = 0.4): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "600 36px 'SF Mono', 'Fira Code', monospace";
  ctx.letterSpacing = "3px";
  ctx.fillStyle = "#2a2a2a";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size * 4, size, 1);
  return sprite;
}

/** Smaller dimension annotation sprite. */
function makeDimLabel(text: string, size = 0.25): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "400 24px 'SF Mono', 'Fira Code', monospace";
  ctx.letterSpacing = "2px";
  ctx.fillStyle = "#999999";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size * 2, size, 1);
  return sprite;
}


export class DebugScene {
  public scene: THREE.Scene;
  private tensorGrids: TensorGrid[] = [];

  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfafafa);

    // --- Lighting (matches PipelineScene) ---
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.4);
    dir.position.set(5, 10, 7);
    this.scene.add(dir);

    // --- Q (8×4) ---
    const qGrid = new TensorGrid(8, 4, { width: 0.8, height: 1.4, depthScale: 1.0, colormap: 'diverging' });
    qGrid.position.set(-1.2, 0.0, 0);
    this.scene.add(qGrid);
    this.tensorGrids.push(qGrid);

    const qLbl = makeLabel("Q", 0.35);
    qLbl.position.set(-1.2, 0.85, 0);
    this.scene.add(qLbl);

    const qDim = makeDimLabel("8×4");
    qDim.position.set(-1.2, -0.85, 0);
    this.scene.add(qDim);

    // --- Kᵀ (4×8, transposed) ---
    const kGrid = new TensorGrid(4, 8, { width: 1.4, height: 0.8, depthScale: 1.0, colormap: 'diverging' });
    kGrid.position.set(0.2, 1.4, 0);
    this.scene.add(kGrid);
    this.tensorGrids.push(kGrid);

    const kLbl = makeLabel("K\u1D40", 0.35);
    kLbl.position.set(0.2, 1.95, 0);
    this.scene.add(kLbl);

    const kDim = makeDimLabel("4×8");
    kDim.position.set(1.15, 1.4, 0);
    this.scene.add(kDim);

    // --- Vᵀ (4×8, transposed) — rotated 90° toward viewer, cube cells ---
    const vRows = 4, vCols = 8, vWidth = 1.4, vHeight = 0.7, vDepthScale = 1.0;
    const vCellGap = 0.85;
    const vCellW = (vWidth / vCols) * vCellGap;
    const vCellH = (vHeight / vRows) * vCellGap;
    const vCubeDepth = Math.min(vCellW, vCellH) * vDepthScale;

    // Attn top-row center Y: attn is 8×8, height=1.4, centered at y=0
    const attnTopRowY = (7 / 2) * (1.4 / 8); // = 0.6125

    // After -π/2 rotation, extrusion goes upward: cube spans [pivotY, pivotY+depth].
    // Align cube center with Attn top row: pivotY + depth/2 = attnTopRowY
    const vPivotY = attnTopRowY - vCubeDepth / 2;

    const vPivot = new THREE.Group();
    vPivot.position.set(0.2, vPivotY, 0);
    vPivot.rotation.x = -Math.PI / 2; // rotate toward viewer

    const vGrid = new TensorGrid(vRows, vCols, { width: vWidth, height: vHeight, depthScale: vDepthScale, colormap: 'diverging' });
    // Offset so top edge sits at pivot origin (grid center is half-height below)
    vGrid.position.set(0, -vHeight / 2, 0);
    vPivot.add(vGrid);

    const vOuter = new THREE.Group();
    vOuter.add(vPivot);
    vOuter.position.set(0, 0, 0.3);
    this.scene.add(vOuter);
    this.tensorGrids.push(vGrid);

    const vLbl = makeLabel("V\u1D40", 0.35);
    vLbl.position.set(0.2, vPivotY, 1.15);
    this.scene.add(vLbl);

    const vDim = makeDimLabel("4×8");
    vDim.position.set(1.15, vPivotY, 0.85);
    this.scene.add(vDim);

    // --- Attention matrix (8×8 TensorGrid in negative space between Q and Kᵀ) ---
    const attnGrid = new TensorGrid(8, 8, { width: 1.4, height: 1.4, depthScale: 1.0, colormap: 'heat' });
    // Generate attention scores: each row sums to 1 (softmax-like)
    const attnData = new Float32Array(64);
    for (let r = 0; r < 8; r++) {
      const raw = new Float32Array(8);
      let sum = 0;
      for (let c = 0; c < 8; c++) {
        raw[c] = Math.exp(Math.random() * 10 - 5);
        sum += raw[c];
      }
      for (let c = 0; c < 8; c++) raw[c] /= sum;
      // Normalize per-row to [0,1] so max cell is always 1.0
      let rMin = raw[0], rMax = raw[0];
      for (let c = 1; c < 8; c++) {
        if (raw[c] < rMin) rMin = raw[c];
        if (raw[c] > rMax) rMax = raw[c];
      }
      const range = rMax - rMin || 1;
      for (let c = 0; c < 8; c++) {
        attnData[r * 8 + c] = (raw[c] - rMin) / range;
      }
    }
    attnGrid.setValues(attnData);
    attnGrid.position.set(0.2, 0.0, 0);
    this.scene.add(attnGrid);
    this.tensorGrids.push(attnGrid);

    const attnLbl = makeLabel("Attn", 0.3);
    attnLbl.position.set(0.2, 0.85, 0);
    this.scene.add(attnLbl);

    const attnDim = makeDimLabel("8×8");
    attnDim.position.set(0.2, -0.85, 0);
    this.scene.add(attnDim);

  }

  /** Per-frame update (no-op — all colors are static). */
  update(_dt: number): void {}
}
