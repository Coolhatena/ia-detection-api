"""Sesiones de validación en memoria.

El backend es la autoridad del paso actual: la app solo renderiza lo que recibe.
Guardarlas en memoria alcanza para el MVP de una LAN; si el proceso reinicia, la
app recibe un 404 tipado y reinicia el flujo limpio.
"""

import time
import uuid
from dataclasses import dataclass, field

from .config import Sequence

SESSION_TTL_SECONDS = 2 * 60 * 60


@dataclass
class Session:
	id: str
	sequence: Sequence
	step_index: int = 0
	created_at: float = field(default_factory=time.monotonic)
	touched_at: float = field(default_factory=time.monotonic)

	@property
	def current_step(self):
		return self.sequence.steps[self.step_index]

	@property
	def is_complete(self) -> bool:
		return self.step_index >= len(self.sequence.steps)


class SessionStore:
	def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS):
		self._sessions: dict[str, Session] = {}
		self._ttl = ttl_seconds

	def create(self, sequence: Sequence) -> Session:
		self._purge_expired()
		session = Session(id=str(uuid.uuid4()), sequence=sequence)
		self._sessions[session.id] = session
		return session

	def get(self, session_id: str) -> Session | None:
		self._purge_expired()
		session = self._sessions.get(session_id)
		if session is not None:
			session.touched_at = time.monotonic()
		return session

	def discard(self, session_id: str) -> bool:
		return self._sessions.pop(session_id, None) is not None

	def _purge_expired(self) -> None:
		cutoff = time.monotonic() - self._ttl
		expired = [sid for sid, s in self._sessions.items() if s.touched_at < cutoff]
		for session_id in expired:
			del self._sessions[session_id]
