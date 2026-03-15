/** Parameter panel — sliders for generation parameters. */

import type { GenerateRequest } from "../types";

export class ParameterPanel {
  private defaults: GenerateRequest = {
    captions: "A gentle piano melody",
    lyrics: "",
    inference_steps: 8,
    guidance_scale: 7.0,
    audio_duration: 10.0,
    use_random_seed: true,
  };

  /** Return current parameter values (uses defaults for now). */
  getParams(): GenerateRequest {
    return { ...this.defaults };
  }
}
