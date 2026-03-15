/**
 * DiT Stage Node — Stacked wireframe planes with diffusion animation.
 *
 * Visual: Stack of horizontal planes (one per transformer layer) on a vertical axis.
 * Each plane is a wireframe grid. Diffusion steps animate as refinement —
 * the grid starts noisy/chaotic and progressively sharpens.
 * Cross-attention connections arc from the LM's output to each layer.
 * Shape: wide and layered.
 */

import * as THREE from "three";
import { PipelineNode } from "./PipelineNode";
import { createWireframeMaterial, setGlobalActivation } from "../rendering/WireframeMaterial";
import type { StageStatus } from "../types";

const LAYER_COUNT = 8;
const GRID_SIZE = 12;
const LAYER_SPACING = 0.7;
const PLANE_SIZE = 3.5;

export class DiTNode extends PipelineNode {
  private layerMeshes: THREE.Mesh[] = [];
  private centralAxis!: THREE.Line;
  private wireMats: THREE.ShaderMaterial[] = [];
  private noiseOffsets: Float32Array[] = [];
  private diffusionProgress = 0; // 0 = fully noisy, 1 = fully clean
  private currentStep = 0;
  private totalSteps = 8;

  constructor() {
    super("DiT");
    this.build();
  }

  private build(): void {
    const stackHeight = (LAYER_COUNT - 1) * LAYER_SPACING;

    // --- Central vertical axis ---
    const axisGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, stackHeight / 2 + 0.5, 0),
      new THREE.Vector3(0, -stackHeight / 2 - 0.5, 0),
    ]);
    this.centralAxis = new THREE.Line(
      axisGeom,
      new THREE.LineBasicMaterial({ color: 0x1a1a1a, linewidth: 1 })
    );
    this.add(this.centralAxis);

    // --- Stacked wireframe planes ---
    for (let l = 0; l < LAYER_COUNT; l++) {
      const y = stackHeight / 2 - l * LAYER_SPACING;
      const mat = createWireframeMaterial({ opacity: 0.5 + l * 0.05 });
      this.wireMats.push(mat);

      const planeGeom = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE, GRID_SIZE, GRID_SIZE);

      // Store noise offsets for this layer
      const posArray = planeGeom.getAttribute("position").array as Float32Array;
      const offsets = new Float32Array(posArray.length);
      for (let i = 0; i < offsets.length; i++) {
        offsets[i] = (Math.random() - 0.5) * 0.4;
      }
      this.noiseOffsets.push(offsets);

      // Add per-vertex activation attribute
      const vertCount = planeGeom.getAttribute("position").count;
      const activations = new Float32Array(vertCount);
      planeGeom.setAttribute("activation", new THREE.BufferAttribute(activations, 1));

      const mesh = new THREE.Mesh(planeGeom, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = y;
      mesh.userData.layerIndex = l;

      this.layerMeshes.push(mesh);
      this.add(mesh);
    }

    // --- Cross-attention arcs (decorative curves from left side) ---
    for (let l = 0; l < LAYER_COUNT; l += 2) {
      const y = stackHeight / 2 - l * LAYER_SPACING;
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(-PLANE_SIZE, y, 0),
        new THREE.Vector3(-PLANE_SIZE * 0.7, y + 0.3, 0.5),
        new THREE.Vector3(-PLANE_SIZE / 2, y, 0),
      );
      const curveGeom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(16));
      const curveLine = new THREE.Line(
        curveGeom,
        new THREE.LineBasicMaterial({ color: 0xd4d4d4, linewidth: 1 })
      );
      this.add(curveLine);
    }

    // --- Label ---
    this.add(this.createLabel("DiT", stackHeight / 2 + 1.0));
  }

  /** Called when a DiT diffusion step event arrives. */
  setDiffusionStep(step: number, total: number): void {
    this.currentStep = step;
    this.totalSteps = total;
    this.diffusionProgress = (step + 1) / total;
  }

  update(dt: number): void {
    // Animate the noisy→clean transition on each layer plane
    for (let l = 0; l < this.layerMeshes.length; l++) {
      const mesh = this.layerMeshes[l];
      const geom = mesh.geometry;
      const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
      const actAttr = geom.getAttribute("activation") as THREE.BufferAttribute;
      const offsets = this.noiseOffsets[l];

      // Noise amplitude decreases as diffusion progresses
      const noiseScale = Math.max(0, 1 - this.diffusionProgress);
      // Layer-dependent phase: deeper layers refine later
      const layerPhase = l / LAYER_COUNT;
      const effectiveNoise = noiseScale * (0.5 + 0.5 * layerPhase);

      for (let i = 0; i < posAttr.count; i++) {
        // Animate z-displacement (since plane is rotated, z = vertical displacement)
        const baseZ = 0;
        const noiseZ = offsets[i * 3 + 2] * effectiveNoise;
        // Add subtle oscillation when active
        const osc =
          this.status === "started"
            ? Math.sin(Date.now() * 0.003 + i * 0.1 + l) * 0.02 * this.activationLevel
            : 0;
        posAttr.setZ(i, baseZ + noiseZ + osc);

        // Activation glow: higher for later diffusion steps, center-biased
        const cx = posAttr.getX(i) / (PLANE_SIZE / 2);
        const cy = posAttr.getY(i) / (PLANE_SIZE / 2);
        const distFromCenter = Math.sqrt(cx * cx + cy * cy);
        const activation = this.activationLevel * this.diffusionProgress * (1 - distFromCenter * 0.5);
        actAttr.setX(i, Math.max(0, activation));
      }
      posAttr.needsUpdate = true;
      actAttr.needsUpdate = true;
    }
  }

  protected onStatusChange(status: StageStatus): void {
    if (status === "idle") {
      this.diffusionProgress = 0;
      this.currentStep = 0;
      this.setActivation(0);
    } else if (status === "started") {
      this.diffusionProgress = 0;
      this.setActivation(0.4);
    } else if (status === "complete") {
      this.diffusionProgress = 1;
      this.setActivation(0.9);
    }
  }

  protected onActivationChange(level: number): void {
    for (const mat of this.wireMats) {
      setGlobalActivation(mat, level * 0.3);
    }
  }
}
