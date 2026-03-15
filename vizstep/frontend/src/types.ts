/** Shared type definitions for VizStep frontend. */

export interface ModelMetadata {
  stages: {
    lm: StageMetadata;
    dit: StageMetadata;
    vae: StageMetadata;
  };
  sample_rate: number;
  device: string;
}

export interface StageMetadata {
  name: string;
  type: string;
  param_count?: number;
  layers?: number;
  vocab_size?: number;
  backend?: string;
  config?: Record<string, unknown>;
}

export interface GenerateRequest {
  captions?: string;
  lyrics?: string;
  inference_steps?: number;
  guidance_scale?: number;
  audio_duration?: number;
  use_random_seed?: boolean;
}

export interface GenerateResponse {
  session_id: string;
  status: string;
}

/** Binary message types matching backend serialization.py */
export const MSG_STAGE_EVENT = 0;
export const MSG_ACTIVATION = 1;
export const MSG_DIT_STEP = 2;
export const MSG_AUDIO_READY = 3;

export interface StageEvent {
  type: typeof MSG_STAGE_EVENT;
  stage: string;
  status: string;
}

export interface ActivationEvent {
  type: typeof MSG_ACTIVATION;
  layerName: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  norm: number;
  heatmap?: Uint8Array;
  heatmapWidth?: number;
  heatmapHeight?: number;
}

export interface DiTStepEvent {
  type: typeof MSG_DIT_STEP;
  step: number;
  total: number;
}

export interface AudioReadyEvent {
  type: typeof MSG_AUDIO_READY;
  sessionId: string;
}

export type VizEvent = StageEvent | ActivationEvent | DiTStepEvent | AudioReadyEvent;

export type StageStatus = "idle" | "started" | "step" | "complete";
