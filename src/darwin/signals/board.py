"""Point-in-time store of intelligence signals consumed by fast agents."""

from __future__ import annotations

from collections import deque

from darwin.core.events import IntelligenceSignal


class SignalBoard:
    """Holds recent :class:`IntelligenceSignal` events.

    ``value()`` only aggregates signals whose availability timestamp is <= ``now`` — even if a
    caller were to insert a future-dated signal, it stays invisible until its time comes.
    """

    def __init__(self, max_signals: int = 2_000, horizon_half_lives: float = 6.0) -> None:
        self._signals: deque[IntelligenceSignal] = deque(maxlen=max_signals)
        self.horizon_half_lives = horizon_half_lives

    def add(self, sig: IntelligenceSignal) -> None:
        self._signals.append(sig)

    def __len__(self) -> int:
        return len(self._signals)

    def recent(self, now: int, limit: int = 50) -> list[IntelligenceSignal]:
        out = [s for s in self._signals if s.ts <= now]
        return out[-limit:]

    def value(self, symbol: str, topic: str, now: int) -> tuple[float, list[str]]:
        num = 0.0
        den = 0.0
        ids: list[str] = []
        for s in self._signals:
            if s.ts > now:
                continue  # not yet available: never leak
            if s.symbol not in (None, symbol):
                continue
            if topic and s.topic != topic:
                continue
            age = now - s.ts
            if age > s.half_life_ms * self.horizon_half_lives:
                continue
            w = s.confidence * 0.5 ** (age / s.half_life_ms)
            if w <= 0:
                continue
            num += w * s.value
            den += w
            ids.append(s.signal_id)
        if den <= 0:
            return 0.0, ids
        # shrink towards 0 when total evidence weight is small
        return num / (den + 0.5), ids
