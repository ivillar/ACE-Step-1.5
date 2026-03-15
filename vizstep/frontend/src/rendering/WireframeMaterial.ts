/** Custom wireframe material with red glow mixing based on activation magnitude. */

import * as THREE from "three";

const vertexShader = /* glsl */ `
  attribute float activation;
  varying float vActivation;
  varying vec3 vPosition;

  void main() {
    vActivation = activation;
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uBaseColor;
  uniform vec3 uActiveColor;
  uniform float uGlobalActivation;
  uniform float uOpacity;
  varying float vActivation;
  varying vec3 vPosition;

  void main() {
    float act = max(vActivation, uGlobalActivation);
    vec3 color = mix(uBaseColor, uActiveColor, act);
    gl_FragColor = vec4(color, uOpacity);
  }
`;

export interface WireframeMaterialOptions {
  baseColor?: THREE.Color;
  activeColor?: THREE.Color;
  opacity?: number;
}

export function createWireframeMaterial(
  opts: WireframeMaterialOptions = {}
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uBaseColor: {
        value: opts.baseColor ?? new THREE.Color(0x1a1a1a),
      },
      uActiveColor: {
        value: opts.activeColor ?? new THREE.Color(0xe63946),
      },
      uGlobalActivation: { value: 0.0 },
      uOpacity: { value: opts.opacity ?? 1.0 },
    },
    transparent: true,
    wireframe: true,
    depthTest: true,
  });
}

export function setGlobalActivation(
  mat: THREE.ShaderMaterial,
  value: number
): void {
  mat.uniforms.uGlobalActivation.value = Math.max(0, Math.min(1, value));
}

// --- Colormap material: per-vertex activation → multi-stop heatmap ---

const colormapFragmentShader = /* glsl */ `
  uniform float uOpacity;
  varying float vActivation;

  vec3 colormap(float t) {
    // Indigo → Blue → Teal → Amber → Red
    vec3 c0 = vec3(0.18, 0.09, 0.45);
    vec3 c1 = vec3(0.11, 0.39, 0.64);
    vec3 c2 = vec3(0.09, 0.59, 0.53);
    vec3 c3 = vec3(0.84, 0.63, 0.14);
    vec3 c4 = vec3(0.78, 0.14, 0.14);

    float s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    if (s < 3.0) return mix(c2, c3, s - 2.0);
    return mix(c3, c4, s - 3.0);
  }

  void main() {
    vec3 color = colormap(vActivation);
    gl_FragColor = vec4(color, uOpacity);
  }
`;

/** CPU-side colormap matching the GLSL 5-stop heatmap (indigo→blue→teal→amber→red). */
export function colormapRGB(t: number): [number, number, number] {
  const c0: [number, number, number] = [0.18, 0.09, 0.45];
  const c1: [number, number, number] = [0.11, 0.39, 0.64];
  const c2: [number, number, number] = [0.09, 0.59, 0.53];
  const c3: [number, number, number] = [0.84, 0.63, 0.14];
  const c4: [number, number, number] = [0.78, 0.14, 0.14];

  const s = Math.max(0, Math.min(1, t)) * 4;
  const lerp = (a: [number, number, number], b: [number, number, number], f: number): [number, number, number] => [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];

  if (s < 1) return lerp(c0, c1, s);
  if (s < 2) return lerp(c1, c2, s - 1);
  if (s < 3) return lerp(c2, c3, s - 2);
  return lerp(c3, c4, s - 3);
}

/** CPU-side diverging colormap (RdBu): maps t ∈ [-1, 1] to blue → white → red. */
export function divergingColormapRGB(t: number): [number, number, number] {
  const c0: [number, number, number] = [0.13, 0.40, 0.67]; // -1.0 deep blue
  const c1: [number, number, number] = [0.57, 0.77, 0.87]; // -0.5 light blue
  const c2: [number, number, number] = [0.97, 0.97, 0.97]; //  0.0 near white
  const c3: [number, number, number] = [0.96, 0.65, 0.51]; // +0.5 salmon
  const c4: [number, number, number] = [0.70, 0.09, 0.17]; // +1.0 deep red

  const s = Math.max(-1, Math.min(1, t)) * 0.5 + 0.5; // remap [-1,1] → [0,1]
  const u = s * 4; // [0,4] for 4 segments
  const lerp = (a: [number, number, number], b: [number, number, number], f: number): [number, number, number] => [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];

  if (u < 1) return lerp(c0, c1, u);
  if (u < 2) return lerp(c1, c2, u - 1);
  if (u < 3) return lerp(c2, c3, u - 2);
  return lerp(c3, c4, u - 3);
}

export function createColormapMaterial(
  opts: { opacity?: number } = {}
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader: colormapFragmentShader,
    uniforms: {
      uOpacity: { value: opts.opacity ?? 1.0 },
    },
    transparent: true,
    wireframe: true,
    depthTest: true,
  });
}

// --- Instanced colormap material: per-instance activation → GPU heatmap ---

export type ColormapMode = 'sequential' | 'diverging' | 'heat';

const instancedColormapVertexShader = /* glsl */ `
  attribute float instanceActivation;
  varying float vActivation;

  void main() {
    vActivation = instanceActivation;
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  }
`;

const instancedColormapFragmentShader = /* glsl */ `
  uniform int uColormapMode; // 0 = sequential, 1 = diverging, 2 = heat

  varying float vActivation;

  vec3 sequentialColormap(float t) {
    vec3 c0 = vec3(0.18, 0.09, 0.45);
    vec3 c1 = vec3(0.11, 0.39, 0.64);
    vec3 c2 = vec3(0.09, 0.59, 0.53);
    vec3 c3 = vec3(0.84, 0.63, 0.14);
    vec3 c4 = vec3(0.78, 0.14, 0.14);

    float s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    if (s < 3.0) return mix(c2, c3, s - 2.0);
    return mix(c3, c4, s - 3.0);
  }

  vec3 divergingColormap(float t) {
    vec3 c0 = vec3(0.10, 0.30, 0.90);
    vec3 c1 = vec3(0.50, 0.72, 0.95);
    vec3 c2 = vec3(0.97, 0.97, 0.97);
    vec3 c3 = vec3(0.98, 0.50, 0.40);
    vec3 c4 = vec3(1.00, 0.09, 0.09);

    float s = clamp(t, -1.0, 1.0) * 0.5 + 0.5;
    float u = s * 4.0;
    if (u < 1.0) return mix(c0, c1, u);
    if (u < 2.0) return mix(c1, c2, u - 1.0);
    if (u < 3.0) return mix(c2, c3, u - 2.0);
    return mix(c3, c4, u - 3.0);
  }

  vec3 heatColormap(float t) {
    // White (0) → Red (1), with pow curve for stronger contrast
    float s = clamp(t, 0.0, 1.0);
    s = s * s; // square for moderate contrast
    return vec3(0.97 + 0.03 * s, 0.97 - 0.88 * s, 0.97 - 0.88 * s);
  }

  void main() {
    vec3 color;
    if (uColormapMode == 2) color = heatColormap(vActivation);
    else if (uColormapMode == 1) color = divergingColormap(vActivation);
    else color = sequentialColormap(vActivation);
    gl_FragColor = vec4(color, 1.0);
  }
`;

export function createInstancedColormapMaterial(
  opts?: { colormap?: ColormapMode }
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: instancedColormapVertexShader,
    fragmentShader: instancedColormapFragmentShader,
    uniforms: {
      uColormapMode: { value: opts?.colormap === 'heat' ? 2 : opts?.colormap === 'diverging' ? 1 : 0 },
    },
    side: THREE.DoubleSide,
    depthTest: true,
  });
}
