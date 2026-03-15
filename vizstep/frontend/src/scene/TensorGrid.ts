/**
 * TensorGrid — Renders a 2D tensor as a colored cell grid using InstancedMesh.
 *
 * Each cell is a small solid-colored plane positioned in a grid with gaps.
 * Colors are computed on the GPU via a multi-stop heatmap shader driven
 * by a per-instance activation attribute.
 */

import * as THREE from "three";
import { ColormapMode, createInstancedColormapMaterial } from "../rendering/WireframeMaterial";

const _dummy = new THREE.Object3D();

export class TensorGrid extends THREE.Group {
  private mesh: THREE.InstancedMesh;
  private rows: number;
  private cols: number;
  private activationAttr: THREE.InstancedBufferAttribute;
  private phases: Float32Array;
  private elapsed = 0;

  constructor(
    rows: number,
    cols: number,
    opts: { width?: number; height?: number; cellGap?: number; colormap?: ColormapMode; depthScale?: number } = {},
  ) {
    super();

    this.rows = rows;
    this.cols = cols;
    const count = rows * cols;

    const width = opts.width ?? 1;
    const height = opts.height ?? 1;
    const gap = opts.cellGap ?? 0.85;

    const cellW = (width / cols) * gap;
    const cellH = (height / rows) * gap;
    const stepX = width / cols;
    const stepY = height / rows;

    // Rounded rectangle extruded for slight depth
    const radius = Math.min(cellW, cellH) * 0.15;
    const depth = Math.min(cellW, cellH) * (opts.depthScale ?? 0.12);
    const shape = new THREE.Shape();
    const hw = cellW / 2;
    const hh = cellH / 2;
    shape.moveTo(-hw + radius, -hh);
    shape.lineTo(hw - radius, -hh);
    shape.quadraticCurveTo(hw, -hh, hw, -hh + radius);
    shape.lineTo(hw, hh - radius);
    shape.quadraticCurveTo(hw, hh, hw - radius, hh);
    shape.lineTo(-hw + radius, hh);
    shape.quadraticCurveTo(-hw, hh, -hw, hh - radius);
    shape.lineTo(-hw, -hh + radius);
    shape.quadraticCurveTo(-hw, -hh, -hw + radius, -hh);
    const bevelSize = radius * 0.6;
    const cellGeom = new THREE.ExtrudeGeometry(shape, {
      depth: depth - bevelSize * 2,
      bevelEnabled: depth > bevelSize * 2,
      bevelThickness: bevelSize,
      bevelSize: bevelSize,
      bevelSegments: 3,
    });

    // Per-instance activation attribute for GPU colormap
    const activationArray = new Float32Array(count);
    this.activationAttr = new THREE.InstancedBufferAttribute(activationArray, 1);
    cellGeom.setAttribute('instanceActivation', this.activationAttr);

    const cellMat = createInstancedColormapMaterial({ colormap: opts.colormap });

    this.mesh = new THREE.InstancedMesh(cellGeom, cellMat, count);
    this.mesh.frustumCulled = false;

    // Position each instance in the grid
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const x = (c - (cols - 1) / 2) * stepX;
        const y = ((rows - 1) / 2 - r) * stepY; // top-to-bottom
        _dummy.position.set(x, y, 0);
        _dummy.updateMatrix();
        this.mesh.setMatrixAt(idx, _dummy.matrix);
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;

    // Initialize activation values
    const isDiverging = opts.colormap === 'diverging';
    this.phases = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      activationArray[i] = isDiverging ? Math.random() * 2 - 1 : Math.random();
      this.phases[i] = Math.random() * Math.PI * 2;
    }
    this.activationAttr.needsUpdate = true;

    this.add(this.mesh);
  }

  /** Replace tensor data and recolor cells. */
  setValues(data: Float32Array): void {
    const arr = this.activationAttr.array as Float32Array;
    arr.set(data.length <= arr.length ? data : data.subarray(0, arr.length));
    if (data.length < arr.length) {
      arr.fill(0, data.length);
    }
    this.activationAttr.needsUpdate = true;
  }

  /** Per-frame update (no-op — colors are static). */
  update(_dt: number): void {}

  private applyColors(): void {
    this.activationAttr.needsUpdate = true;
  }
}
