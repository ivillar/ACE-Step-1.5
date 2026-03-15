"""FastAPI app: REST + WebSocket + static file serving for vizstep."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from vizstep.backend.pipeline_bridge import VizSession, get_model_metadata
from vizstep.backend.session import SessionStore

# Directory paths
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(title="VizStep — ACE-Step Visualization Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — set by the entry point before server starts
_dit_wrapper: Any = None
_lm_wrapper: Any = None
_debug_mode: str | None = None
_sessions = SessionStore()


def configure(dit_wrapper: Any, lm_wrapper: Any = None, debug: str | None = None) -> None:
    """Inject model wrappers into the server module."""
    global _dit_wrapper, _lm_wrapper, _debug_mode
    _dit_wrapper = dit_wrapper
    _lm_wrapper = lm_wrapper
    _debug_mode = debug


# ---------- REST endpoints ----------


@app.get("/api/model/metadata")
async def model_metadata() -> JSONResponse:
    """Return layer counts, dimensions, and param counts per stage."""
    if _dit_wrapper is None:
        if _debug_mode is not None:
            return JSONResponse({"debug": _debug_mode})
        return JSONResponse(
            {"error": "Models not loaded. Start vizstep with a model checkpoint."},
            status_code=503,
        )
    meta = get_model_metadata(_dit_wrapper, _lm_wrapper)
    meta["debug"] = _debug_mode
    return JSONResponse(meta)


@app.post("/api/generate")
async def start_generation(body: dict[str, Any] | None = None) -> JSONResponse:
    """Start a music generation session, returns session_id."""
    if _dit_wrapper is None:
        return JSONResponse({"error": "Models not loaded"}, status_code=503)

    session = _sessions.create()
    params = body or {}

    # Default generation parameters
    gen_params: dict[str, Any] = {
        "captions": params.get("captions", "A gentle piano melody"),
        "lyrics": params.get("lyrics", ""),
        "inference_steps": params.get("inference_steps", 8),
        "guidance_scale": params.get("guidance_scale", 7.0),
        "audio_duration": params.get("audio_duration", 10.0),
        "use_random_seed": params.get("use_random_seed", True),
    }

    # Run generation in background task
    viz_session = VizSession(_dit_wrapper, _lm_wrapper, session)

    async def _run() -> None:
        await viz_session.run_generation(gen_params)

    asyncio.create_task(_run())

    return JSONResponse({"session_id": session.session_id, "status": "started"})


@app.get("/api/audio/{session_id}", response_model=None)
async def get_audio(session_id: str) -> FileResponse | JSONResponse:
    """Serve the generated WAV file for a completed session."""
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if session.audio_path is None or not session.audio_path.exists():
        return JSONResponse({"error": "Audio not ready"}, status_code=404)
    return FileResponse(session.audio_path, media_type="audio/wav", filename=f"{session_id}.wav")


@app.get("/api/sessions")
async def list_sessions() -> JSONResponse:
    """List active session IDs."""
    return JSONResponse({"sessions": _sessions.list_ids()})


# ---------- WebSocket ----------


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Push activation events to the connected frontend client."""
    session = _sessions.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"[ws] Client connected for session {session_id}")

    try:
        while True:
            # Wait for data from the generation pipeline
            try:
                data = await asyncio.wait_for(session.ws_queue.get(), timeout=0.1)
                await websocket.send_bytes(data)
            except TimeoutError:
                # Check if session is done
                if session.status.value in ("complete", "error"):
                    # Drain remaining items
                    while not session.ws_queue.empty():
                        data = session.ws_queue.get_nowait()
                        await websocket.send_bytes(data)
                    break
                # Send a keepalive ping
                continue
    except WebSocketDisconnect:
        logger.info(f"[ws] Client disconnected from session {session_id}")
    except Exception as exc:
        logger.warning(f"[ws] Error in session {session_id}: {exc}")
    finally:
        await websocket.close()


# ---------- Static files (frontend) ----------

@app.on_event("startup")
async def _mount_frontend() -> None:
    """Mount the built frontend if the dist directory exists."""
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
        logger.info(f"[server] Serving frontend from {FRONTEND_DIST}")
    else:
        logger.info("[server] No frontend build found; API-only mode")


def create_app() -> FastAPI:
    """Return the configured FastAPI app (for uvicorn programmatic use)."""
    return app
