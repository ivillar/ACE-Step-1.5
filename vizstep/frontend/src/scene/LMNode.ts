/**
 * LM Stage Node — Transformer-style vertical layer stack (bbycroft-inspired).
 *
 * Visual: Vertical stack of transformer layers, each containing distinct
 * Attention and MLP wireframe sub-blocks with residual arcs. Tokens flow
 * top-to-bottom through the structure.
 *
 * Architecture (Qwen3): Embedding → N × (Attention + MLP) → RMSNorm → LM Head
 */

import * as THREE from "three";
import { PipelineNode } from "./PipelineNode";
import { createWireframeMaterial, setGlobalActivation } from "../rendering/WireframeMaterial";
import type { StageStatus } from "../types";

// --- Layout constants ---
const NUM_LAYERS = 28;         // Qwen3-0.6B; overridden by metadata
const MAX_VISIBLE_LAYERS = 8;  // Compress if numLayers > 12
const COMPRESS_THRESHOLD = 12;

const LAYER_HEIGHT = 0.3;
const LAYER_GAP = 0.08;
const ATTN_WIDTH = 2.5;
const MLP_WIDTH = 3.2;
const BLOCK_DEPTH = 0.8;
const EMBED_WIDTH = 2.5;
const EMBED_HEIGHT = 0.15;
const NORM_HEIGHT = 0.08;
const HEAD_HEIGHT = 0.2;
const HEAD_WIDTH = 2.0;
const TOKEN_SIZE = 0.08;
const CONNECTOR_GAP = 0.06;
const SKIP_INDICATOR_DOTS = 3;
const SKIP_DOT_GAP = 0.08;

// Colors
const COT_COLOR = new THREE.Color(0xd4d4d4);
const CODE_COLOR = new THREE.Color(0xe63946);
const DARK_GRAY = 0x1a1a1a;
const DIVIDER_COLOR = 0x888888;

// Token row
const MAX_TOKENS = 32;
const COT_TOKEN_COUNT = 10;

export class LMNode extends PipelineNode {
  private numLayers: number;
  private visibleLayerCount: number;
  private compressed: boolean;

  // Materials (one per layer so we can color independently)
  private layerMaterials: THREE.ShaderMaterial[] = [];
  private embedMat!: THREE.ShaderMaterial;
  private normMat!: THREE.ShaderMaterial;
  private headMat!: THREE.ShaderMaterial;
  private lineMat!: THREE.LineBasicMaterial;

  // Layer groups for per-layer activation
  private layerGroups: THREE.Group[] = [];

  // Token embedding row
  private tokenMarkers: THREE.Mesh[] = [];
  private tokenRow!: THREE.Group;
  private dividerLine!: THREE.Line;

  // Animation state
  private wavefrontY = 0;
  private wavefrontActive = false;
  private tokensGenerated = 0;
  private tokenTimer = 0;
  private totalHeight = 0;
  private topY = 0;

  constructor(numLayers: number = NUM_LAYERS) {
    super("LM");
    this.numLayers = numLayers;
    this.compressed = numLayers > COMPRESS_THRESHOLD;
    this.visibleLayerCount = this.compressed ? MAX_VISIBLE_LAYERS : numLayers;
    this.build();
  }

  private build(): void {
    this.lineMat = new THREE.LineBasicMaterial({ color: DARK_GRAY, linewidth: 1 });

    // Calculate total height
    const layerBlockHeight = LAYER_HEIGHT * 2 + LAYER_GAP; // attn + mlp
    const skipIndicatorHeight = SKIP_DOT_GAP * (SKIP_INDICATOR_DOTS + 1);
    let stackHeight: number;
    if (this.compressed) {
      // Each visible layer + skip indicators between groups
      const skipCount = this.visibleLayerCount - 1;
      stackHeight = this.visibleLayerCount * layerBlockHeight
        + skipCount * skipIndicatorHeight;
    } else {
      stackHeight = this.visibleLayerCount * layerBlockHeight
        + (this.visibleLayerCount - 1) * CONNECTOR_GAP;
    }

    this.totalHeight = EMBED_HEIGHT + CONNECTOR_GAP
      + stackHeight + CONNECTOR_GAP
      + NORM_HEIGHT + CONNECTOR_GAP
      + HEAD_HEIGHT;

    this.topY = this.totalHeight / 2;
    let cursorY = this.topY;

    // === Token Embedding Row ===
    cursorY = this.buildEmbeddingRow(cursorY);

    // Connector
    cursorY = this.addConnector(cursorY, CONNECTOR_GAP);

    // === Transformer Layer Stack ===
    cursorY = this.buildLayerStack(cursorY);

    // Connector
    cursorY = this.addConnector(cursorY, CONNECTOR_GAP);

    // === Final RMS Norm ===
    cursorY = this.buildNormBar(cursorY);

    // Connector
    cursorY = this.addConnector(cursorY, CONNECTOR_GAP);

    // === LM Head ===
    cursorY = this.buildLMHead(cursorY);

    // === Label ===
    this.add(this.createLabel("LM", this.topY + 0.5));
  }

  // ─── Embedding Row ───────────────────────────────────────────

  private buildEmbeddingRow(startY: number): number {
    this.embedMat = createWireframeMaterial({ opacity: 0.6 });
    const boxGeom = new THREE.BoxGeometry(EMBED_WIDTH, EMBED_HEIGHT, BLOCK_DEPTH);
    const edgesGeom = new THREE.EdgesGeometry(boxGeom);
    this.addActivationAttribute(edgesGeom);
    const embedBox = new THREE.LineSegments(edgesGeom, this.embedMat);
    embedBox.position.set(0, startY - EMBED_HEIGHT / 2, 0);
    this.add(embedBox);

    // Token marker row (hidden initially)
    this.tokenRow = new THREE.Group();
    this.tokenRow.position.set(0, startY - EMBED_HEIGHT / 2, 0);
    this.add(this.tokenRow);

    const tokenGeom = new THREE.BoxGeometry(TOKEN_SIZE, TOKEN_SIZE, TOKEN_SIZE);
    const rowWidth = EMBED_WIDTH * 0.85;
    for (let i = 0; i < MAX_TOKENS; i++) {
      const t = i / (MAX_TOKENS - 1);
      const x = -rowWidth / 2 + t * rowWidth;
      const mat = new THREE.MeshBasicMaterial({
        color: i < COT_TOKEN_COUNT ? COT_COLOR : CODE_COLOR,
        transparent: true,
        opacity: 0,
      });
      const marker = new THREE.Mesh(tokenGeom, mat);
      marker.position.set(x, 0, BLOCK_DEPTH / 2 + TOKEN_SIZE);
      this.tokenRow.add(marker);
      this.tokenMarkers.push(marker);
    }

    // Divider line between CoT and code tokens (hidden initially)
    const dividerX = -rowWidth / 2 + (COT_TOKEN_COUNT / (MAX_TOKENS - 1)) * rowWidth;
    const divGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(dividerX, -EMBED_HEIGHT, BLOCK_DEPTH / 2 + TOKEN_SIZE),
      new THREE.Vector3(dividerX, EMBED_HEIGHT, BLOCK_DEPTH / 2 + TOKEN_SIZE),
    ]);
    this.dividerLine = new THREE.Line(
      divGeom,
      new THREE.LineBasicMaterial({ color: DIVIDER_COLOR, transparent: true, opacity: 0 })
    );
    this.tokenRow.add(this.dividerLine);

    return startY - EMBED_HEIGHT;
  }

  // ─── Transformer Layer Stack ─────────────────────────────────

  private buildLayerStack(startY: number): number {
    let cursorY = startY;

    for (let i = 0; i < this.visibleLayerCount; i++) {
      const layerGroup = new THREE.Group();
      const layerMat = createWireframeMaterial({ opacity: 0.7 });
      this.layerMaterials.push(layerMat);

      // --- Attention block ---
      const attnGeom = new THREE.BoxGeometry(ATTN_WIDTH, LAYER_HEIGHT, BLOCK_DEPTH);
      const attnEdges = new THREE.EdgesGeometry(attnGeom);
      this.addActivationAttribute(attnEdges);
      const attnBox = new THREE.LineSegments(attnEdges, layerMat);
      attnBox.position.set(0, cursorY - LAYER_HEIGHT / 2, 0);
      layerGroup.add(attnBox);

      cursorY -= LAYER_HEIGHT;

      // Small connector between attn and mlp
      const innerConn = this.createConnectorLine(cursorY, LAYER_GAP);
      layerGroup.add(innerConn);
      cursorY -= LAYER_GAP;

      // --- MLP block (wider) ---
      const mlpGeom = new THREE.BoxGeometry(MLP_WIDTH, LAYER_HEIGHT, BLOCK_DEPTH);
      const mlpEdges = new THREE.EdgesGeometry(mlpGeom);
      this.addActivationAttribute(mlpEdges);
      const mlpBox = new THREE.LineSegments(mlpEdges, layerMat);
      mlpBox.position.set(0, cursorY - LAYER_HEIGHT / 2, 0);
      layerGroup.add(mlpBox);

      cursorY -= LAYER_HEIGHT;

      // --- Residual arc (right side) ---
      const residualArc = this.createResidualArc(
        cursorY + LAYER_HEIGHT * 2 + LAYER_GAP, // top of attn
        cursorY,                                   // bottom of mlp
        Math.max(ATTN_WIDTH, MLP_WIDTH) / 2 + 0.15
      );
      layerGroup.add(residualArc);

      this.add(layerGroup);
      this.layerGroups.push(layerGroup);

      // Inter-layer gap
      if (i < this.visibleLayerCount - 1) {
        if (this.compressed) {
          cursorY = this.addSkipIndicator(cursorY);
        } else {
          cursorY = this.addConnector(cursorY, CONNECTOR_GAP);
        }
      }
    }

    return cursorY;
  }

  // ─── RMS Norm Bar ────────────────────────────────────────────

  private buildNormBar(startY: number): number {
    this.normMat = createWireframeMaterial({ opacity: 0.5 });
    const normGeom = new THREE.BoxGeometry(ATTN_WIDTH, NORM_HEIGHT, BLOCK_DEPTH);
    const normEdges = new THREE.EdgesGeometry(normGeom);
    this.addActivationAttribute(normEdges);
    const normBox = new THREE.LineSegments(normEdges, this.normMat);
    normBox.position.set(0, startY - NORM_HEIGHT / 2, 0);
    this.add(normBox);
    return startY - NORM_HEIGHT;
  }

  // ─── LM Head ─────────────────────────────────────────────────

  private buildLMHead(startY: number): number {
    this.headMat = createWireframeMaterial({ opacity: 0.6 });
    const headGeom = new THREE.BoxGeometry(HEAD_WIDTH, HEAD_HEIGHT, BLOCK_DEPTH);
    const headEdges = new THREE.EdgesGeometry(headGeom);
    this.addActivationAttribute(headEdges);
    const headBox = new THREE.LineSegments(headEdges, this.headMat);
    headBox.position.set(0, startY - HEAD_HEIGHT / 2, 0);
    this.add(headBox);
    return startY - HEAD_HEIGHT;
  }

  // ─── Helpers ─────────────────────────────────────────────────

  private addActivationAttribute(geom: THREE.BufferGeometry): void {
    const count = geom.getAttribute("position").count;
    geom.setAttribute("activation", new THREE.BufferAttribute(new Float32Array(count), 1));
  }

  private createConnectorLine(topY: number, length: number): THREE.Line {
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, topY, 0),
      new THREE.Vector3(0, topY - length, 0),
    ]);
    return new THREE.Line(geom, this.lineMat);
  }

  private addConnector(cursorY: number, gap: number): number {
    const line = this.createConnectorLine(cursorY, gap);
    this.add(line);
    return cursorY - gap;
  }

  private addSkipIndicator(cursorY: number): number {
    const totalH = SKIP_DOT_GAP * (SKIP_INDICATOR_DOTS + 1);
    const dotGeom = new THREE.SphereGeometry(0.02, 4, 4);
    const dotMat = new THREE.MeshBasicMaterial({ color: DARK_GRAY });

    for (let d = 0; d < SKIP_INDICATOR_DOTS; d++) {
      const dot = new THREE.Mesh(dotGeom, dotMat);
      dot.position.set(0, cursorY - SKIP_DOT_GAP * (d + 1), 0);
      this.add(dot);
    }
    return cursorY - totalH;
  }

  private createResidualArc(topY: number, bottomY: number, xOffset: number): THREE.Line {
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(xOffset, topY, 0),
      new THREE.Vector3(xOffset + 0.3, (topY + bottomY) / 2, 0),
      new THREE.Vector3(xOffset, bottomY, 0)
    );
    const points = curve.getPoints(16);
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: DARK_GRAY,
      linewidth: 1,
      transparent: true,
      opacity: 0.4,
    });
    return new THREE.Line(geom, mat);
  }

  // ─── Animation ───────────────────────────────────────────────

  update(dt: number): void {
    if (this.status === "idle") return;

    const isGenerating = this.status === "started" || this.status === "step";

    if (isGenerating) {
      // Token generation: fill markers left-to-right
      this.tokenTimer += dt;
      const tokensPerSec = 12;
      const targetTokens = Math.min(MAX_TOKENS, Math.floor(this.tokenTimer * tokensPerSec));

      if (targetTokens > this.tokensGenerated) {
        for (let i = this.tokensGenerated; i < targetTokens; i++) {
          const mat = this.tokenMarkers[i].material as THREE.MeshBasicMaterial;
          mat.opacity = 1.0;
        }
        this.tokensGenerated = targetTokens;
      }

      // Show divider once code tokens start
      if (this.tokensGenerated > COT_TOKEN_COUNT) {
        const divMat = this.dividerLine.material as THREE.LineBasicMaterial;
        divMat.opacity = 0.8;
      }

      // Wavefront sweeping top-to-bottom through layers
      this.wavefrontActive = true;
      const sweepSpeed = this.totalHeight * 0.6; // traverse in ~1.7s
      this.wavefrontY += dt * sweepSpeed;
      if (this.wavefrontY > this.totalHeight) {
        this.wavefrontY = 0; // loop
      }

      // Compute wavefront position in world-Y
      const wavefrontWorldY = this.topY - this.wavefrontY;
      const wavefrontRadius = this.totalHeight * 0.12;

      // Activate layers based on wavefront proximity
      for (let i = 0; i < this.visibleLayerCount; i++) {
        const layerGroup = this.layerGroups[i];
        // Estimate layer center Y from its children (average of attn and mlp)
        const attnBox = layerGroup.children[0] as THREE.LineSegments;
        const mlpBox = layerGroup.children[2] as THREE.LineSegments;
        const layerCenterY = (attnBox.position.y + mlpBox.position.y) / 2;

        const dist = Math.abs(layerCenterY - wavefrontWorldY);
        const intensity = Math.max(0, 1 - dist / wavefrontRadius);
        const activation = intensity * this.activationLevel;
        setGlobalActivation(this.layerMaterials[i], activation);
      }

      // Activate embedding / norm / head based on wavefront
      const embedDist = Math.abs(this.topY - EMBED_HEIGHT / 2 - wavefrontWorldY);
      setGlobalActivation(this.embedMat, Math.max(0, 1 - embedDist / wavefrontRadius) * this.activationLevel);

      const bottomY = this.topY - this.totalHeight;
      const normY = bottomY + HEAD_HEIGHT + CONNECTOR_GAP + NORM_HEIGHT / 2;
      const normDist = Math.abs(normY - wavefrontWorldY);
      setGlobalActivation(this.normMat, Math.max(0, 1 - normDist / wavefrontRadius) * this.activationLevel);

      const headY = bottomY + HEAD_HEIGHT / 2;
      const headDist = Math.abs(headY - wavefrontWorldY);
      setGlobalActivation(this.headMat, Math.max(0, 1 - headDist / wavefrontRadius) * this.activationLevel);
    }

    if (this.status === "complete") {
      // Brief flash then settle
      const flashLevel = this.activationLevel * 0.3;
      for (const mat of this.layerMaterials) {
        setGlobalActivation(mat, flashLevel);
      }
      setGlobalActivation(this.embedMat, flashLevel);
      setGlobalActivation(this.normMat, flashLevel);
      setGlobalActivation(this.headMat, flashLevel);
    }
  }

  protected onStatusChange(status: StageStatus): void {
    if (status === "idle") {
      this.resetAnimation();
      this.setActivation(0);
    } else if (status === "started") {
      this.setActivation(0.5);
    } else if (status === "step") {
      this.setActivation(0.7);
    } else if (status === "complete") {
      this.setActivation(0.8);
    }
  }

  protected onActivationChange(level: number): void {
    // Activation is applied per-layer in update() via wavefront
    // For global fallback (e.g., complete state), set all layers
    if (this.status === "complete" || this.status === "idle") {
      const val = level * 0.4;
      for (const mat of this.layerMaterials) {
        setGlobalActivation(mat, val);
      }
      setGlobalActivation(this.embedMat, val);
      setGlobalActivation(this.normMat, val);
      setGlobalActivation(this.headMat, val);
    }
  }

  private resetAnimation(): void {
    this.wavefrontY = 0;
    this.wavefrontActive = false;
    this.tokensGenerated = 0;
    this.tokenTimer = 0;

    // Hide all token markers
    for (const marker of this.tokenMarkers) {
      (marker.material as THREE.MeshBasicMaterial).opacity = 0;
    }

    // Hide divider
    (this.dividerLine.material as THREE.LineBasicMaterial).opacity = 0;

    // Reset all layer activations
    for (const mat of this.layerMaterials) {
      setGlobalActivation(mat, 0);
    }
    setGlobalActivation(this.embedMat, 0);
    setGlobalActivation(this.normMat, 0);
    setGlobalActivation(this.headMat, 0);
  }
}
