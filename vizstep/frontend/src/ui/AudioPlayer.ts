/** Inline audio playback — plays generated WAV when pipeline completes. */

import { audioUrl } from "../network/RestClient";

export class AudioPlayer {
  private container: HTMLElement;
  private audioEl: HTMLAudioElement | null = null;

  constructor(containerId: string) {
    this.container = document.getElementById(containerId)!;
  }

  /** Load and display audio controls for a completed session. */
  play(sessionId: string): void {
    this.clear();
    this.audioEl = document.createElement("audio");
    this.audioEl.controls = true;
    this.audioEl.src = audioUrl(sessionId);
    this.audioEl.autoplay = true;
    this.container.appendChild(this.audioEl);
  }

  clear(): void {
    if (this.audioEl) {
      this.audioEl.pause();
      this.audioEl.remove();
      this.audioEl = null;
    }
  }
}
