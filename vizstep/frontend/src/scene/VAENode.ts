/**
 * VAE Stage Node — Hourglass funnel with waveform output.
 *
 * Visual: Wide input mesh compresses to a narrow latent bottleneck,
 * then expands to the output waveform. On decode, energy visually flows
 * through the funnel. The output end displays the audio waveform as
 * a line oscilloscope.
 */

import * as THREE from "three";
import { PipelineNode } from "./PipelineNode";
import { createWireframeMaterial, setGlobalActivation } from "../rendering/WireframeMaterial";
import type { StageStatus } from "../types";

const FUNNEL_HEIGHT = 5;
const TOP_RADIUS = 2.0;
const NECK_RADIUS = 0.4;
const BOTTOM_RADIUS = 1.8;
const SEGMENTS = 24;
const RINGS = 12;
const WAVEFORM_POINTS = 128;

export class VAENode extends PipelineNode {
  private encoderFunnel!: THREE.Mesh;
  private decoderFunnel!: THREE.Mesh;
  private waveformLine!: THREE.Line;
  private wireMat!: THREE.ShaderMaterial;
  private energyRings: THREE.Line[] = [];
  private energyProgress = 0;
  private waveformData: Float32Array;

  constructor() {
    super("VAE");
    this.waveformData = new Float32Array(WAVEFORM_POINTS);
    this.build();
  }

  private build(): void {
    this.wireMat = createWireframeMaterial({ opacity: 0.5 });

    const halfH = FUNNEL_HEIGHT / 2;
    const neckY = 0;

    // --- Encoder funnel (top half: wide → narrow) ---
    const encGeom = this.createFunnelGeometry(TOP_RADIUS, NECK_RADIUS, halfH, RINGS / 2);
    this.encoderFunnel = new THREE.Mesh(encGeom, this.wireMat);
    this.encoderFunnel.position.y = halfH / 2;
    this.add(this.encoderFunnel);

    // --- Decoder funnel (bottom half: narrow → wide) ---
    const decGeom = this.createFunnelGeometry(NECK_RADIUS, BOTTOM_RADIUS, halfH, RINGS / 2);
    this.decoderFunnel = new THREE.Mesh(decGeom, this.wireMat);
    this.decoderFunnel.position.y = -halfH / 2;
    this.add(this.decoderFunnel);

    // --- Bottleneck ring ---
    const ringGeom = new THREE.RingGeometry(NECK_RADIUS * 0.8, NECK_RADIUS * 1.2, SEGMENTS);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xe63946,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.3,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = neckY;
    this.add(ring);

    // --- Energy flow rings (animated) ---
    for (let i = 0; i < 5; i++) {
      const r = NECK_RADIUS + (TOP_RADIUS - NECK_RADIUS) * (i / 5);
      const eGeom = new THREE.RingGeometry(r * 0.95, r * 1.05, SEGMENTS);
      const eMat = new THREE.LineBasicMaterial({
        color: 0xe63946,
        transparent: true,
        opacity: 0,
      });
      const eLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(
          new THREE.Path(new THREE.EllipseCurve(0, 0, r, r, 0, Math.PI * 2, false, 0).getPoints(SEGMENTS)).getPoints()
        ),
        eMat,
      );
      eLine.rotation.x = Math.PI / 2;
      this.energyRings.push(eLine);
      this.add(eLine);
    }

    // --- Waveform output line (oscilloscope) ---
    const wavePositions = new Float32Array(WAVEFORM_POINTS * 3);
    for (let i = 0; i < WAVEFORM_POINTS; i++) {
      wavePositions[i * 3] = (i / WAVEFORM_POINTS - 0.5) * BOTTOM_RADIUS * 2;
      wavePositions[i * 3 + 1] = -FUNNEL_HEIGHT / 2 - 0.8;
      wavePositions[i * 3 + 2] = 0;
    }
    const waveGeom = new THREE.BufferGeometry();
    waveGeom.setAttribute("position", new THREE.BufferAttribute(wavePositions, 3));
    this.waveformLine = new THREE.Line(
      waveGeom,
      new THREE.LineBasicMaterial({ color: 0xe63946, linewidth: 1.5 })
    );
    this.add(this.waveformLine);

    // --- Label ---
    this.add(this.createLabel("VAE", FUNNEL_HEIGHT / 2 + 0.5));
  }

  private createFunnelGeometry(
    topR: number,
    bottomR: number,
    height: number,
    rings: number
  ): THREE.BufferGeometry {
    // Build a cylinder-like shape with varying radii (lathe geometry)
    const points: THREE.Vector2[] = [];
    for (let i = 0; i <= rings; i++) {
      const t = i / rings;
      const r = topR + (bottomR - topR) * t;
      const y = (0.5 - t) * height;
      points.push(new THREE.Vector2(r, y));
    }
    const geom = new THREE.LatheGeometry(points, SEGMENTS);

    // Add activation attribute
    const count = geom.getAttribute("position").count;
    geom.setAttribute("activation", new THREE.BufferAttribute(new Float32Array(count), 1));

    return geom;
  }

  update(dt: number): void {
    if (this.status === "started" || this.status === "step") {
      // Animate energy flowing through the funnel
      this.energyProgress += dt * 2;
      if (this.energyProgress > 1) this.energyProgress -= 1;

      for (let i = 0; i < this.energyRings.length; i++) {
        const phase = (i / this.energyRings.length + this.energyProgress) % 1;
        const y = FUNNEL_HEIGHT / 2 - phase * FUNNEL_HEIGHT;
        this.energyRings[i].position.y = y;
        const mat = this.energyRings[i].material as THREE.LineBasicMaterial;
        // Fade in at top, full in middle, fade at bottom
        const fadeIn = Math.min(1, phase * 4);
        const fadeOut = Math.min(1, (1 - phase) * 4);
        mat.opacity = fadeIn * fadeOut * this.activationLevel * 0.8;
      }
    }

    if (this.status === "complete") {
      // Animate the waveform
      const posAttr = this.waveformLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      const time = Date.now() * 0.002;
      for (let i = 0; i < WAVEFORM_POINTS; i++) {
        const x = i / WAVEFORM_POINTS;
        const y =
          Math.sin(x * 20 + time) * 0.15 +
          Math.sin(x * 35 + time * 1.3) * 0.08 +
          Math.sin(x * 8 + time * 0.7) * 0.1;
        posAttr.setY(i, -FUNNEL_HEIGHT / 2 - 0.8 + y);
      }
      posAttr.needsUpdate = true;
    }
  }

  protected onStatusChange(status: StageStatus): void {
    if (status === "idle") {
      this.energyProgress = 0;
      this.setActivation(0);
      // Reset waveform
      const posAttr = this.waveformLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < WAVEFORM_POINTS; i++) {
        posAttr.setY(i, -FUNNEL_HEIGHT / 2 - 0.8);
      }
      posAttr.needsUpdate = true;
    } else if (status === "started") {
      this.setActivation(0.4);
    } else if (status === "complete") {
      this.setActivation(0.7);
    }
  }

  protected onActivationChange(level: number): void {
    setGlobalActivation(this.wireMat, level * 0.3);
  }
}
