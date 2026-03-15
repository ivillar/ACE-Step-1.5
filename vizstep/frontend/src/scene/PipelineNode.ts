/** Abstract base class for stage nodes in the pipeline visualization. */

import * as THREE from "three";
import type { StageStatus, ActivationEvent } from "../types";

export abstract class PipelineNode extends THREE.Group {
  public stageName: string;
  public status: StageStatus = "idle";
  protected activationLevel = 0;

  constructor(name: string) {
    super();
    this.stageName = name;
    this.name = name;
  }

  /** Set the stage status and trigger visual transitions. */
  setStatus(status: StageStatus): void {
    this.status = status;
    this.onStatusChange(status);
  }

  /** Update activation level (0-1) for glow effects. */
  setActivation(level: number): void {
    this.activationLevel = Math.max(0, Math.min(1, level));
    this.onActivationChange(this.activationLevel);
  }

  /** Handle an incoming activation event from WebSocket. */
  handleActivation(event: ActivationEvent): void {
    // Normalize activation to 0-1 based on norm
    const normalized = Math.min(1, Math.abs(event.norm) / 100);
    this.setActivation(normalized);
  }

  /** Per-frame update. dt is in seconds. */
  abstract update(dt: number): void;

  /** Called when stage status changes. */
  protected abstract onStatusChange(status: StageStatus): void;

  /** Called when activation level changes. */
  protected abstract onActivationChange(level: number): void;

  /** Build a wireframe label for the stage. */
  protected createLabel(text: string, y: number): THREE.Group {
    const group = new THREE.Group();
    // Use a simple point marker since Three.js text requires font loading
    const dotGeom = new THREE.SphereGeometry(0.06, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
    const dot = new THREE.Mesh(dotGeom, dotMat);
    dot.position.set(0, y, 0);
    group.add(dot);
    group.userData.label = text;
    return group;
  }
}
