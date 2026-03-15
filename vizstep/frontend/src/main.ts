/**
 * VizStep Main — Scene bootstrap, OrbitControls, camera, lighting, animation loop.
 *
 * Connects the 3D pipeline scene to the WebSocket backend and UI controls.
 * Features: smooth fly-in camera on load, click-to-inspect raycasting,
 * and live generation with stage-by-stage animation.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PipelineScene } from "./scene/PipelineScene";
import { DebugScene } from "./scene/DebugScene";
import { WebSocketClient } from "./network/WebSocketClient";
import { fetchModelMetadata, startGeneration } from "./network/RestClient";
import { AudioPlayer } from "./ui/AudioPlayer";
import { ParameterPanel } from "./ui/ParameterPanel";
import { MSG_STAGE_EVENT, MSG_AUDIO_READY } from "./types";
import type { VizEvent, StageEvent, AudioReadyEvent } from "./types";

// --- DOM Elements ---
const container = document.getElementById("canvas-container")!;
const statusStage = document.getElementById("status-stage")!;
const btnGenerate = document.getElementById("btn-generate")! as HTMLButtonElement;
const detailOverlay = document.getElementById("detail-overlay")!;

// --- Renderer ---
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xfafafa);
container.appendChild(renderer.domElement);

// --- Camera ---
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  200
);
// Start far away for fly-in animation
camera.position.set(0, 20, 30);
camera.lookAt(0, 0, 0);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);
controls.maxDistance = 50;
controls.minDistance = 3;

// --- Pipeline Scene ---
const pipelineScene = new PipelineScene();

// --- Networking ---
const wsClient = new WebSocketClient();
const audioPlayer = new AudioPlayer("audio-player");
const paramPanel = new ParameterPanel();

// --- Raycaster for click-to-inspect ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// --- State ---
let isGenerating = false;
let flyInProgress = 0; // 0 = far, 1 = arrived
const FLY_IN_DURATION = 2.5; // seconds
const TARGET_POS = new THREE.Vector3(0, 5, 16);

// --- Fly-in camera animation ---
const startCamPos = new THREE.Vector3(0, 20, 30);

function updateFlyIn(dt: number): void {
  if (flyInProgress >= 1) return;
  flyInProgress = Math.min(1, flyInProgress + dt / FLY_IN_DURATION);

  // Smooth ease-out
  const t = 1 - Math.pow(1 - flyInProgress, 3);
  camera.position.lerpVectors(startCamPos, TARGET_POS, t);
  camera.lookAt(0, 0, 0);
  controls.target.set(0, 0, 0);
}

// --- Event handling ---
function handleVizEvent(event: VizEvent): void {
  // Forward to pipeline scene
  pipelineScene.handleEvent(event);

  // Update status bar
  if (event.type === MSG_STAGE_EVENT) {
    const se = event as StageEvent;
    statusStage.textContent = `${se.stage.toUpperCase()} ${se.status.toUpperCase()}`;

    if (se.stage === "vae" && se.status === "complete") {
      statusStage.textContent = "COMPLETE";
    }
  }

  // Play audio when ready
  if (event.type === MSG_AUDIO_READY) {
    const ae = event as AudioReadyEvent;
    audioPlayer.play(ae.sessionId);
    isGenerating = false;
    btnGenerate.disabled = false;
  }
}

// --- Generate button ---
btnGenerate.addEventListener("click", async () => {
  if (isGenerating) return;
  isGenerating = true;
  btnGenerate.disabled = true;
  statusStage.textContent = "STARTING";

  pipelineScene.reset();
  audioPlayer.clear();

  try {
    const params = paramPanel.getParams();
    const response = await startGeneration(params);

    // Connect WebSocket for this session
    wsClient.disconnect();
    wsClient.onEvent(handleVizEvent);
    wsClient.connect(response.session_id);
  } catch (err) {
    console.error("Generation failed:", err);
    statusStage.textContent = "ERROR";
    isGenerating = false;
    btnGenerate.disabled = false;
  }
});

// --- Click-to-inspect ---
renderer.domElement.addEventListener("click", (event) => {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(pipelineScene.scene.children, true);

  if (intersects.length > 0) {
    // Walk up to find the PipelineNode
    let obj: THREE.Object3D | null = intersects[0].object;
    while (obj && !obj.userData.label && obj.parent) {
      obj = obj.parent;
    }

    if (obj && obj.userData.label) {
      detailOverlay.style.display = "block";
      detailOverlay.innerHTML = `
        <div class="label">${obj.userData.label}</div>
        <div>Status: ${(obj as any).status || "N/A"}</div>
      `;
      // Auto-hide after 3 seconds
      setTimeout(() => {
        detailOverlay.style.display = "none";
      }, 3000);
    }
  } else {
    detailOverlay.style.display = "none";
  }
});

// --- Resize handler ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Debug mode support ---
let debugScene: DebugScene | null = null;

// --- Load model metadata on startup ---
fetchModelMetadata()
  .then((meta: any) => {
    console.log("Model metadata:", meta);

    if (meta.debug === "attn") {
      // Debug mode: render a single attention block
      debugScene = new DebugScene();
      camera.position.set(0, 0.5, 6);
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      flyInProgress = 1; // skip fly-in
      statusStage.textContent = "DEBUG: attn";
      return;
    }

    statusStage.textContent = "READY";
  })
  .catch(() => {
    console.log("Running in standalone mode (no backend)");
    statusStage.textContent = "DEMO";

    // Run a demo animation cycle
    runDemoAnimation();
  });

function runDemoAnimation(): void {
  // Simulate a generation cycle for visual testing without backend
  const stages = [
    { stage: "lm", status: "started", delay: 1000 },
    { stage: "lm", status: "complete", delay: 3000 },
    { stage: "dit", status: "started", delay: 3500 },
    { stage: "dit", status: "step", delay: 4000 },
    { stage: "dit", status: "complete", delay: 7000 },
    { stage: "vae", status: "started", delay: 7500 },
    { stage: "vae", status: "complete", delay: 9000 },
  ];

  for (const s of stages) {
    setTimeout(() => {
      pipelineScene.handleEvent({
        type: MSG_STAGE_EVENT,
        stage: s.stage,
        status: s.status,
      });
      statusStage.textContent = `${s.stage.toUpperCase()} ${s.status.toUpperCase()}`;
    }, s.delay);
  }

  // DiT diffusion steps
  for (let step = 0; step < 8; step++) {
    setTimeout(
      () => {
        pipelineScene.ditNode.setDiffusionStep(step, 8);
      },
      3500 + step * 400
    );
  }

  // Loop the demo
  setTimeout(() => {
    pipelineScene.reset();
    statusStage.textContent = "DEMO";
    setTimeout(runDemoAnimation, 2000);
  }, 12000);
}

// --- Animation loop ---
let lastTime = performance.now();

function animate(): void {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  updateFlyIn(dt);
  controls.update();

  if (debugScene) {
    debugScene.update(dt);
    renderer.render(debugScene.scene, camera);
  } else {
    pipelineScene.update(dt);
    renderer.render(pipelineScene.scene, camera);
  }
}

animate();
