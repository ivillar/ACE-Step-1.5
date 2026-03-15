"""Per-client generation session state."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SessionStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Session:
    """Tracks the state of a single visualization session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: SessionStatus = SessionStatus.IDLE
    current_stage: str | None = None
    audio_path: Path | None = None
    ws_queue: asyncio.Queue | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ws_queue is None:
            self.ws_queue = asyncio.Queue()


class SessionStore:
    """Thread-safe store for active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())
