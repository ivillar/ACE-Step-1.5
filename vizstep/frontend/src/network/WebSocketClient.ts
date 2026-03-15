/** WebSocket client for binary activation streaming. */

import {
  MSG_STAGE_EVENT,
  MSG_ACTIVATION,
  MSG_DIT_STEP,
  MSG_AUDIO_READY,
  type VizEvent,
  type StageEvent,
  type ActivationEvent,
  type DiTStepEvent,
  type AudioReadyEvent,
} from "../types";

export type EventCallback = (event: VizEvent) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private listeners: EventCallback[] = [];

  connect(sessionId: string): void {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}/ws/${sessionId}`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onmessage = (ev) => {
      const data = new DataView(ev.data as ArrayBuffer);
      const event = this.decode(data);
      if (event) {
        for (const cb of this.listeners) cb(event);
      }
    };

    this.ws.onclose = () => {
      this.ws = null;
    };
  }

  onEvent(cb: EventCallback): void {
    this.listeners.push(cb);
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  private decode(view: DataView): VizEvent | null {
    const type = view.getUint8(0);

    switch (type) {
      case MSG_STAGE_EVENT:
        return this.decodeStageEvent(view);
      case MSG_ACTIVATION:
        return this.decodeActivation(view);
      case MSG_DIT_STEP:
        return this.decodeDiTStep(view);
      case MSG_AUDIO_READY:
        return this.decodeAudioReady(view);
      default:
        return null;
    }
  }

  private decodeStageEvent(view: DataView): StageEvent {
    let offset = 1;
    const stageLen = view.getUint8(offset++);
    const stageBytes = new Uint8Array(view.buffer, offset, stageLen);
    const stage = new TextDecoder().decode(stageBytes);
    offset += stageLen;
    const statusLen = view.getUint8(offset++);
    const statusBytes = new Uint8Array(view.buffer, offset, statusLen);
    const status = new TextDecoder().decode(statusBytes);
    return { type: MSG_STAGE_EVENT, stage, status };
  }

  private decodeActivation(view: DataView): ActivationEvent {
    let offset = 1;
    const nameLen = view.getUint16(offset);
    offset += 2;
    const nameBytes = new Uint8Array(view.buffer, offset, nameLen);
    const layerName = new TextDecoder().decode(nameBytes);
    offset += nameLen;

    const mean = view.getFloat32(offset);
    offset += 4;
    const std = view.getFloat32(offset);
    offset += 4;
    const min = view.getFloat32(offset);
    offset += 4;
    const max = view.getFloat32(offset);
    offset += 4;
    const norm = view.getFloat32(offset);
    offset += 4;

    const hasHeatmap = view.getUint8(offset++);
    const event: ActivationEvent = {
      type: MSG_ACTIVATION,
      layerName,
      mean,
      std,
      min,
      max,
      norm,
    };

    if (hasHeatmap) {
      event.heatmapHeight = view.getUint16(offset);
      offset += 2;
      event.heatmapWidth = view.getUint16(offset);
      offset += 2;
      // Remaining bytes are zlib-compressed heatmap data
      // For now, skip decompression in the browser; use raw stats
    }

    return event;
  }

  private decodeDiTStep(view: DataView): DiTStepEvent {
    return {
      type: MSG_DIT_STEP,
      step: view.getUint16(1),
      total: view.getUint16(3),
    };
  }

  private decodeAudioReady(view: DataView): AudioReadyEvent {
    const sidLen = view.getUint8(1);
    const sidBytes = new Uint8Array(view.buffer, 2, sidLen);
    return {
      type: MSG_AUDIO_READY,
      sessionId: new TextDecoder().decode(sidBytes),
    };
  }
}
