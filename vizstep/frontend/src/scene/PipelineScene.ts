/**
 * PipelineScene — Top-level scene that positions three stage nodes
 * (LM → DiT → VAE) in 3D space with data flow particle streams.
 */

import * as THREE from "three";
import { LMNode } from "./LMNode";
import { DiTNode } from "./DiTNode";
import { VAENode } from "./VAENode";
import { DataFlow } from "./DataFlow";
import type { VizEvent, StageEvent, ActivationEvent, DiTStepEvent } from "../types";
import { MSG_STAGE_EVENT, MSG_ACTIVATION, MSG_DIT_STEP } from "../types";

// Stage positions along the X axis (left to right: LM → DiT → VAE)
const LM_POS = new THREE.Vector3(-7, 0, 0);
const DIT_POS = new THREE.Vector3(0, 0, 0);
const VAE_POS = new THREE.Vector3(7, 0, 0);

// Data flow connection points (output face → input face)
const LM_OUT = new THREE.Vector3(-5.2, 0, 0);
const DIT_IN = new THREE.Vector3(-2, 0, 0);
const DIT_OUT = new THREE.Vector3(2, 0, 0);
const VAE_IN = new THREE.Vector3(5.5, 0, 0);

export class PipelineScene {
  public scene: THREE.Scene;
  public lmNode: LMNode;
  public ditNode: DiTNode;
  public vaeNode: VAENode;
  public dataFlow: DataFlow;

  private lmToDitStream: number;
  private ditToVaeStream: number;

  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfafafa);

    // --- Stage nodes ---
    this.lmNode = new LMNode();
    this.lmNode.position.copy(LM_POS);
    this.scene.add(this.lmNode);

    this.ditNode = new DiTNode();
    this.ditNode.position.copy(DIT_POS);
    this.scene.add(this.ditNode);

    this.vaeNode = new VAENode();
    this.vaeNode.position.copy(VAE_POS);
    this.scene.add(this.vaeNode);

    // --- Data flow streams ---
    this.dataFlow = new DataFlow();
    this.scene.add(this.dataFlow);

    this.lmToDitStream = this.dataFlow.addStream(LM_OUT, DIT_IN);
    this.ditToVaeStream = this.dataFlow.addStream(DIT_OUT, VAE_IN);

    // --- Ground reference grid ---
    const gridHelper = new THREE.GridHelper(30, 30, 0xd4d4d4, 0xd4d4d4);
    gridHelper.position.y = -4;
    gridHelper.material.transparent = true;
    (gridHelper.material as THREE.Material).opacity = 0.3;
    this.scene.add(gridHelper);

    // --- Subtle ambient + directional lighting ---
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.4);
    directional.position.set(5, 10, 7);
    this.scene.add(directional);
  }

  /** Handle incoming visualization events from WebSocket. */
  handleEvent(event: VizEvent): void {
    switch (event.type) {
      case MSG_STAGE_EVENT:
        this.handleStageEvent(event as StageEvent);
        break;
      case MSG_ACTIVATION:
        this.handleActivation(event as ActivationEvent);
        break;
      case MSG_DIT_STEP:
        this.handleDiTStep(event as DiTStepEvent);
        break;
    }
  }

  private handleStageEvent(event: StageEvent): void {
    const node = this.getNode(event.stage);
    if (node) {
      node.setStatus(event.status as "idle" | "started" | "step" | "complete");
    }

    // Activate/deactivate data flow streams based on stage transitions
    if (event.stage === "lm" && event.status === "started") {
      this.dataFlow.setStreamActive(this.lmToDitStream, false);
      this.dataFlow.setStreamActive(this.ditToVaeStream, false);
    } else if (event.stage === "lm" && event.status === "complete") {
      this.dataFlow.setStreamActive(this.lmToDitStream, true);
    } else if (event.stage === "dit" && event.status === "complete") {
      this.dataFlow.setStreamActive(this.lmToDitStream, false);
      this.dataFlow.setStreamActive(this.ditToVaeStream, true);
    } else if (event.stage === "vae" && event.status === "complete") {
      this.dataFlow.setStreamActive(this.ditToVaeStream, false);
    }
  }

  private handleActivation(event: ActivationEvent): void {
    // Route activation to the appropriate node based on layer name prefix
    const parts = event.layerName.split("/");
    const stage = parts[0];
    const node = this.getNode(stage);
    if (node) {
      node.handleActivation(event);
    }
  }

  private handleDiTStep(event: DiTStepEvent): void {
    this.ditNode.setDiffusionStep(event.step, event.total);
  }

  private getNode(stage: string) {
    switch (stage) {
      case "lm":
        return this.lmNode;
      case "dit":
        return this.ditNode;
      case "vae":
        return this.vaeNode;
      default:
        return null;
    }
  }

  /** Per-frame update. dt is in seconds. */
  update(dt: number): void {
    this.lmNode.update(dt);
    this.ditNode.update(dt);
    this.vaeNode.update(dt);
    this.dataFlow.update(dt);
  }

  /** Reset all stages to idle. */
  reset(): void {
    this.lmNode.setStatus("idle");
    this.ditNode.setStatus("idle");
    this.vaeNode.setStatus("idle");
    this.dataFlow.setAllActive(false);
  }
}
