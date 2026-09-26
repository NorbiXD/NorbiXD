"""The scheduling seam shared by the replay and live drivers."""

from __future__ import annotations

from typing import Protocol

from darwin.core.events import Event


class EventTarget(Protocol):
    def handle(self, event: Event) -> None: ...


class Scheduler(Protocol):
    """Deliver ``event`` to ``target`` at time ``ts`` (ms).

    Replay: a deterministic priority queue ordered by ``(ts, priority, seq)``.
    Live: a wall-clock timer heap feeding the engine's single consumer loop.
    """

    def schedule(self, ts: int, event: Event, target: EventTarget) -> None: ...
