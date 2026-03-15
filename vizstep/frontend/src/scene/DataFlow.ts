/**
 * DataFlow — Particle streams between stages with concentric ring effects.
 *
 * Particles travel along thin wireframe edges from one stage's output face
 * to the next stage's input. Connection points have concentric ring ripple
 * animations. Particle density reflects data volume.
 */

import * as THREE from "three";

const PARTICLES_PER_STREAM = 40;
const RING_COUNT = 3;

interface ParticleStream {
  points: THREE.Points;
  positions: Float32Array;
  velocities: Float32Array;
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  active: boolean;
}

interface ConnectionRing {
  meshes: THREE.Mesh[];
  position: THREE.Vector3;
  active: boolean;
  time: number;
}

export class DataFlow extends THREE.Group {
  private streams: ParticleStream[] = [];
  private rings: ConnectionRing[] = [];

  constructor() {
    super();
    this.name = "DataFlow";
  }

  /** Add a particle stream between two 3D positions. */
  addStream(from: THREE.Vector3, to: THREE.Vector3): number {
    const positions = new Float32Array(PARTICLES_PER_STREAM * 3);
    const velocities = new Float32Array(PARTICLES_PER_STREAM);
    const sizes = new Float32Array(PARTICLES_PER_STREAM);

    for (let i = 0; i < PARTICLES_PER_STREAM; i++) {
      // Distribute particles along the path with random offset
      const t = Math.random();
      const pos = new THREE.Vector3().lerpVectors(from, to, t);
      // Add slight perpendicular scatter
      pos.x += (Math.random() - 0.5) * 0.15;
      pos.z += (Math.random() - 0.5) * 0.15;
      positions[i * 3] = pos.x;
      positions[i * 3 + 1] = pos.y;
      positions[i * 3 + 2] = pos.z;
      velocities[i] = 0.3 + Math.random() * 0.4;
      sizes[i] = 1.0 + Math.random() * 1.5;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute(
      "size",
      new THREE.BufferAttribute(sizes, 1)
    );

    const mat = new THREE.PointsMaterial({
      color: 0xe63946,
      size: 0.05,
      transparent: true,
      opacity: 0,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geom, mat);
    this.add(points);

    const stream: ParticleStream = {
      points,
      positions,
      velocities,
      startPos: from.clone(),
      endPos: to.clone(),
      active: false,
    };
    this.streams.push(stream);

    // Add connection rings at endpoints
    this.addConnectionRing(from);
    this.addConnectionRing(to);

    return this.streams.length - 1;
  }

  /** Add concentric ring ripple effect at a connection point. */
  private addConnectionRing(pos: THREE.Vector3): void {
    const meshes: THREE.Mesh[] = [];

    for (let r = 0; r < RING_COUNT; r++) {
      const radius = 0.2 + r * 0.15;
      const geom = new THREE.RingGeometry(radius * 0.9, radius * 1.1, 32);
      const mat = new THREE.MeshBasicMaterial({
        color: 0xe63946,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(pos);
      mesh.rotation.x = Math.PI / 2;
      meshes.push(mesh);
      this.add(mesh);
    }

    this.rings.push({ meshes, position: pos, active: false, time: 0 });
  }

  /** Activate/deactivate a stream by index. */
  setStreamActive(index: number, active: boolean): void {
    if (index >= 0 && index < this.streams.length) {
      this.streams[index].active = active;
    }
    // Activate corresponding rings (2 rings per stream)
    const ringStart = index * 2;
    for (let r = ringStart; r < ringStart + 2 && r < this.rings.length; r++) {
      this.rings[r].active = active;
      this.rings[r].time = 0;
    }
  }

  /** Activate all streams. */
  setAllActive(active: boolean): void {
    for (let i = 0; i < this.streams.length; i++) {
      this.setStreamActive(i, active);
    }
  }

  update(dt: number): void {
    const direction = new THREE.Vector3();

    for (const stream of this.streams) {
      const mat = stream.points.material as THREE.PointsMaterial;

      if (!stream.active) {
        // Fade out
        mat.opacity = Math.max(0, mat.opacity - dt * 2);
        continue;
      }

      // Fade in
      mat.opacity = Math.min(0.7, mat.opacity + dt * 2);

      // Move particles along the stream direction
      direction.subVectors(stream.endPos, stream.startPos).normalize();
      const totalDist = stream.startPos.distanceTo(stream.endPos);

      const posAttr = stream.points.geometry.getAttribute("position") as THREE.BufferAttribute;

      for (let i = 0; i < PARTICLES_PER_STREAM; i++) {
        const px = posAttr.getX(i);
        const py = posAttr.getY(i);
        const pz = posAttr.getZ(i);

        // Project particle onto stream axis to find progress
        const toParticle = new THREE.Vector3(px, py, pz).sub(stream.startPos);
        let progress = toParticle.dot(direction) / totalDist;

        progress += stream.velocities[i] * dt;

        if (progress > 1) {
          // Reset to start with scatter
          progress = Math.random() * 0.1;
        }

        const newPos = new THREE.Vector3().lerpVectors(stream.startPos, stream.endPos, progress);
        newPos.x += (Math.random() - 0.5) * 0.05;
        newPos.z += (Math.random() - 0.5) * 0.05;

        posAttr.setXYZ(i, newPos.x, newPos.y, newPos.z);
      }
      posAttr.needsUpdate = true;
    }

    // Animate connection rings
    for (const ring of this.rings) {
      if (!ring.active) {
        for (const mesh of ring.meshes) {
          (mesh.material as THREE.MeshBasicMaterial).opacity = Math.max(
            0,
            ((mesh.material as THREE.MeshBasicMaterial).opacity ?? 0) - dt * 2
          );
        }
        continue;
      }

      ring.time += dt;
      for (let r = 0; r < ring.meshes.length; r++) {
        const phase = (ring.time * 1.5 + r * 0.3) % 1;
        const mat = ring.meshes[r].material as THREE.MeshBasicMaterial;
        mat.opacity = Math.sin(phase * Math.PI) * 0.4;
        const scale = 1 + phase * 0.3;
        ring.meshes[r].scale.set(scale, scale, scale);
      }
    }
  }
}
