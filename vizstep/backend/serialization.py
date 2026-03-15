"""Binary frame encoding for activation summaries sent over WebSocket."""

from __future__ import annotations

import struct
import zlib

import numpy as np

# Binary frame layout:
# [1 byte]  message type  (0=stage_event, 1=activation_summary, 2=dit_step, 3=audio_ready)
# Payload varies by type.

MSG_STAGE_EVENT = 0
MSG_ACTIVATION = 1
MSG_DIT_STEP = 2
MSG_AUDIO_READY = 3


def encode_stage_event(stage: str, status: str) -> bytes:
    """Encode a stage transition event.

    Layout: [type:1][stage_len:1][stage:N][status_len:1][status:N]
    """
    stage_bytes = stage.encode("utf-8")
    status_bytes = status.encode("utf-8")
    return struct.pack("!BB", MSG_STAGE_EVENT, len(stage_bytes)) + stage_bytes + struct.pack("!B", len(status_bytes)) + status_bytes


def encode_dit_step(step: int, total: int) -> bytes:
    """Encode a DiT diffusion step counter.

    Layout: [type:1][step:2][total:2]
    """
    return struct.pack("!BHH", MSG_DIT_STEP, step, total)


def encode_audio_ready(session_id: str) -> bytes:
    """Encode an audio-ready notification.

    Layout: [type:1][session_id_len:1][session_id:N]
    """
    sid_bytes = session_id.encode("utf-8")
    return struct.pack("!BB", MSG_AUDIO_READY, len(sid_bytes)) + sid_bytes


def encode_activation_summary(
    layer_name: str,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
    norm: float,
    heatmap: np.ndarray | None = None,
) -> bytes:
    """Encode an activation summary with optional downsampled heatmap.

    Layout:
      [type:1][name_len:2][name:N][mean:4][std:4][min:4][max:4][norm:4]
      [has_heatmap:1]
      if has_heatmap:
        [h:2][w:2][compressed_data:...]
    """
    name_bytes = layer_name.encode("utf-8")
    header = struct.pack("!BH", MSG_ACTIVATION, len(name_bytes)) + name_bytes
    stats = struct.pack("!fffff", mean, std, min_val, max_val, norm)

    if heatmap is not None:
        heatmap_u8 = heatmap.astype(np.uint8)
        h, w = heatmap_u8.shape[:2]
        compressed = zlib.compress(heatmap_u8.tobytes(), level=1)
        heatmap_data = struct.pack("!BHH", 1, h, w) + compressed
    else:
        heatmap_data = struct.pack("!B", 0)

    return header + stats + heatmap_data


def decode_message_type(data: bytes) -> int:
    """Read the message type byte from a binary frame."""
    return data[0]
