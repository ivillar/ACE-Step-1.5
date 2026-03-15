/** Custom point cloud material with sized/colored points for neuron populations. */

import * as THREE from "three";

const vertexShader = /* glsl */ `
  attribute float activation;
  attribute float size;
  varying float vActivation;

  uniform float uBaseSize;
  uniform float uActiveSizeScale;

  void main() {
    vActivation = activation;
    float s = size * uBaseSize * (1.0 + activation * uActiveSizeScale);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = s * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uBaseColor;
  uniform vec3 uActiveColor;
  uniform float uGlobalActivation;
  varying float vActivation;

  void main() {
    // Circular point
    vec2 center = gl_PointCoord - 0.5;
    float dist = length(center);
    if (dist > 0.5) discard;

    float act = max(vActivation, uGlobalActivation);
    vec3 color = mix(uBaseColor, uActiveColor, act);

    // Soft edge
    float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
    gl_FragColor = vec4(color, alpha);
  }
`;

export interface PointCloudMaterialOptions {
  baseColor?: THREE.Color;
  activeColor?: THREE.Color;
  baseSize?: number;
  activeSizeScale?: number;
}

export function createPointCloudMaterial(
  opts: PointCloudMaterialOptions = {}
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
      uBaseSize: { value: opts.baseSize ?? 3.0 },
      uActiveSizeScale: { value: opts.activeSizeScale ?? 2.0 },
    },
    transparent: true,
    depthTest: true,
  });
}
