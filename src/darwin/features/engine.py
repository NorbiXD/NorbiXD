"""Incremental feature engine.

The engine keeps a bounded history of *closed* bars per symbol. At each decision tick it hands
agents a :class:`FeatureView` — a read-only, memoised window over that history. Two properties
matter:

1. **No look-ahead by construction.** A view only contains bars whose ``end_ts <= now`` and
   intelligence signals whose availability ``ts <= now``. There is no API to reach further.
2. **Attribution by construction.** Every indicator an agent reads is recorded in
   ``view.accessed`` with its value, so each decision persists exactly what the agent "saw".
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from darwin.core.types import Regime
from darwin.market.bars import Bar

if TYPE_CHECKING:
    from darwin.signals.board import SignalBoard

FloatArray = npt.NDArray[np.float64]
NAN = float("nan")


@dataclass
class _Arrays:
    close: FloatArray
    high: FloatArray
    low: FloatArray
    volume: FloatArray
    buy: FloatArray
    sell: FloatArray
    funding: FloatArray
    oi: FloatArray
    liq_long: FloatArray
    liq_short: FloatArray
    notional: FloatArray


def _arrays(bars: list[Bar]) -> _Arrays:
    return _Arrays(
        close=np.fromiter((b.close for b in bars), float, len(bars)),
        high=np.fromiter((b.high for b in bars), float, len(bars)),
        low=np.fromiter((b.low for b in bars), float, len(bars)),
        volume=np.fromiter((b.volume for b in bars), float, len(bars)),
        buy=np.fromiter((b.buy_volume for b in bars), float, len(bars)),
        sell=np.fromiter((b.sell_volume for b in bars), float, len(bars)),
        funding=np.fromiter((b.funding_rate for b in bars), float, len(bars)),
        oi=np.fromiter((b.open_interest for b in bars), float, len(bars)),
        liq_long=np.fromiter((b.liq_long_notional for b in bars), float, len(bars)),
        liq_short=np.fromiter((b.liq_short_notional for b in bars), float, len(bars)),
        notional=np.fromiter((b.volume * b.vwap for b in bars), float, len(bars)),
    )


@dataclass
class FeatureView:
    """Read-only memoised indicator window for one symbol at one decision time."""

    symbol: str
    now: int
    bars: list[Bar]
    signals: SignalBoard | None = None
    accessed: dict[str, float] = field(default_factory=dict)
    signals_seen: set[str] = field(default_factory=set)
    _cache: dict[str, float] = field(default_factory=dict)
    _arr: _Arrays | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def a(self) -> _Arrays:
        if self._arr is None:
            self._arr = _arrays(self.bars)
        return self._arr

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def last(self) -> Bar:
        return self.bars[-1]

    @property
    def stale(self) -> bool:
        return bool(self.bars) and self.bars[-1].stale

    def ready(self, n: int) -> bool:
        return len(self.bars) > n

    def _memo(self, key: str, fn: Callable[[], float]) -> float:
        v = self._cache.get(key)
        if v is None:
            v = float(fn())
            self._cache[key] = v
        self.accessed[key] = v
        return v

    def fresh(self) -> FeatureView:
        """A view sharing history and cache but with an empty access log (one per agent)."""
        return FeatureView(
            symbol=self.symbol,
            now=self.now,
            bars=self.bars,
            signals=self.signals,
            _cache=self._cache,
            _arr=self.a,
        )

    # ------------------------------------------------------------------ price
    def close(self) -> float:
        return self._memo("close", lambda: self.bars[-1].close)

    def ret(self, n: int) -> float:
        """Log return over the last ``n`` bars."""

        def f() -> float:
            c = self.a.close
            if len(c) <= n:
                return NAN
            return math.log(c[-1] / c[-1 - n])

        return self._memo(f"ret({n})", f)

    def vol(self, n: int) -> float:
        """Std-dev of 1-bar log returns over the last ``n`` bars."""

        def f() -> float:
            c = self.a.close
            if len(c) <= max(n, 2):
                return NAN
            r = np.diff(np.log(c[-n - 1 :]))
            s = float(np.std(r, ddof=1))
            return s if s > 0 else NAN

        return self._memo(f"vol({n})", f)

    def zret(self, n: int, vol_n: int | None = None) -> float:
        """Return over ``n`` bars normalised by volatility: ret(n) / (vol * sqrt(n))."""

        def f() -> float:
            r = self.ret(n)
            v = self.vol(vol_n or max(n, 20))
            if not (math.isfinite(r) and math.isfinite(v)):
                return NAN
            return r / (v * math.sqrt(n))

        return self._memo(f"zret({n},{vol_n})", f)

    def zprice(self, n: int) -> float:
        """(close - SMA(n)) / std(close, n)."""

        def f() -> float:
            c = self.a.close
            if len(c) < n or n < 3:
                return NAN
            w = c[-n:]
            sd = float(np.std(w, ddof=1))
            if sd <= 0:
                return NAN
            return float((c[-1] - float(np.mean(w))) / sd)

        return self._memo(f"zprice({n})", f)

    def rsi(self, n: int) -> float:
        def f() -> float:
            c = self.a.close
            if len(c) <= n:
                return NAN
            d = np.diff(c[-n - 1 :])
            up = float(d[d > 0].sum())
            dn = float(-d[d < 0].sum())
            if up + dn <= 0:
                return 50.0
            return 100.0 * up / (up + dn)

        return self._memo(f"rsi({n})", f)

    def channel_position(self, n: int) -> float:
        """Where the current close sits vs. the high/low channel of the *previous* ``n`` bars.

        > 1 means a breakout above the channel, < 0 a breakdown below it.
        """

        def f() -> float:
            a = self.a
            if len(a.close) <= n:
                return NAN
            hi = float(np.max(a.high[-n - 1 : -1]))
            lo = float(np.min(a.low[-n - 1 : -1]))
            if hi <= lo:
                return NAN
            return float((a.close[-1] - lo) / (hi - lo))

        return self._memo(f"channel_position({n})", f)

    def atr_pct(self, n: int) -> float:
        def f() -> float:
            a = self.a
            if len(a.close) <= n:
                return NAN
            h, lo, c = a.high[-n:], a.low[-n:], a.close[-n - 1 : -1]
            tr = np.maximum(h - lo, np.maximum(np.abs(h - c), np.abs(lo - c)))
            return float(np.mean(tr)) / float(a.close[-1])

        return self._memo(f"atr_pct({n})", f)

    def vol_ratio(self, short: int, long: int) -> float:
        def f() -> float:
            s, lg = self.vol(short), self.vol(long)
            if not (math.isfinite(s) and math.isfinite(lg)) or lg <= 0:
                return NAN
            return s / lg

        return self._memo(f"vol_ratio({short},{long})", f)

    # ------------------------------------------------------------------ flow / book
    def flow_imbalance(self, n: int) -> float:
        """Taker buy minus taker sell volume share over ``n`` bars, in [-1, 1]."""

        def f() -> float:
            a = self.a
            if len(a.buy) < n:
                return NAN
            b, s = float(a.buy[-n:].sum()), float(a.sell[-n:].sum())
            return (b - s) / (b + s) if b + s > 0 else 0.0

        return self._memo(f"flow_imbalance({n})", f)

    def book_imbalance(self) -> float:
        return self._memo("book_imbalance", lambda: self.bars[-1].book_imbalance)

    def spread_bps(self) -> float:
        return self._memo("spread_bps", lambda: self.bars[-1].spread_bps)

    # ------------------------------------------------------------------ derivatives data
    def funding(self) -> float:
        return self._memo("funding", lambda: self.bars[-1].funding_rate)

    def funding_z(self, n: int) -> float:
        def f() -> float:
            fr = self.a.funding
            if len(fr) < n or n < 3:
                return NAN
            w = fr[-n:]
            sd = float(np.std(w, ddof=1))
            if sd <= 1e-12:
                return 0.0
            return float((fr[-1] - float(np.mean(w))) / sd)

        return self._memo(f"funding_z({n})", f)

    def oi_change(self, n: int) -> float:
        def f() -> float:
            oi = self.a.oi
            if len(oi) <= n or oi[-1 - n] <= 0 or oi[-1] <= 0:
                return NAN
            return math.log(oi[-1] / oi[-1 - n])

        return self._memo(f"oi_change({n})", f)

    def liq_imbalance(self, n: int) -> float:
        """(shorts liquidated - longs liquidated) / total over ``n`` bars. +1 = short squeeze."""

        def f() -> float:
            a = self.a
            if len(a.liq_long) < n:
                return NAN
            lo, sh = float(a.liq_long[-n:].sum()), float(a.liq_short[-n:].sum())
            return (sh - lo) / (sh + lo) if sh + lo > 0 else 0.0

        return self._memo(f"liq_imbalance({n})", f)

    def liq_intensity(self, n: int, baseline: int = 120) -> float:
        """Liquidation notional over ``n`` bars relative to average traded notional per bar."""

        def f() -> float:
            a = self.a
            if len(a.notional) < n:
                return NAN
            liq = float(a.liq_long[-n:].sum() + a.liq_short[-n:].sum())
            base = float(np.mean(a.notional[-baseline:])) if len(a.notional) else 0.0
            if base <= 0:
                return 0.0
            return liq / (base * n)

        return self._memo(f"liq_intensity({n},{baseline})", f)

    # ------------------------------------------------------------------ intelligence
    def signal(self, topic: str = "") -> float:
        """Decay-weighted directional intelligence score for this symbol in [-1, 1]."""

        def f() -> float:
            if self.signals is None:
                return 0.0
            value, ids = self.signals.value(self.symbol, topic, self.now)
            self.signals_seen.update(ids)
            return value

        key = f"signal({topic})"
        v = f()  # not memoised across agents: we need per-agent signals_seen
        self.accessed[key] = v
        return v

    # ------------------------------------------------------------------ regime
    def regime(self) -> Regime:
        v = self._memo("regime_code", self._regime_code)
        return _REGIMES[int(v)]

    def _regime_code(self) -> float:
        c = self.a.close
        if len(c) < 62:
            return float(_REGIMES.index(Regime.UNKNOWN))
        vr = self.vol_ratio(30, min(240, len(c) - 1))
        if math.isfinite(vr) and vr > 1.8:
            return float(_REGIMES.index(Regime.HIGH_VOL))
        w = c[-61:]
        path = float(np.abs(np.diff(w)).sum())
        er = abs(float(w[-1] - w[0])) / path if path > 0 else 0.0
        if er > 0.30:
            return float(_REGIMES.index(Regime.TREND_UP if w[-1] > w[0] else Regime.TREND_DOWN))
        return float(_REGIMES.index(Regime.RANGE))


_REGIMES: list[Regime] = [Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE, Regime.HIGH_VOL, Regime.UNKNOWN]


_FIELDS = (
    "close",
    "high",
    "low",
    "volume",
    "buy",
    "sell",
    "funding",
    "oi",
    "liq_long",
    "liq_short",
    "notional",
    "end_ts",
)


def _bar_values(b: Bar) -> tuple[float, ...]:
    return (
        b.close,
        b.high,
        b.low,
        b.volume,
        b.buy_volume,
        b.sell_volume,
        b.funding_rate,
        b.open_interest,
        b.liq_long_notional,
        b.liq_short_notional,
        b.volume * b.vwap,
        float(b.end_ts),
    )


class _SymbolBuffer:
    """Append-only columnar bar history. Slices handed to views stay valid after appends
    (compaction allocates a fresh buffer instead of overwriting in place)."""

    def __init__(self, max_bars: int) -> None:
        self.max_bars = max_bars
        self.cap = max_bars * 4
        self.data = np.zeros((len(_FIELDS), self.cap))
        self.start = 0
        self.n = 0
        self.bars: deque[Bar] = deque(maxlen=max_bars)

    def append(self, bar: Bar) -> None:
        if self.start + self.n >= self.cap:
            keep = min(self.n, self.max_bars - 1)
            fresh = np.zeros_like(self.data)
            fresh[:, :keep] = self.data[:, self.start + self.n - keep : self.start + self.n]
            self.data, self.start, self.n = fresh, 0, keep
        self.data[:, self.start + self.n] = _bar_values(bar)
        self.n += 1
        if self.n > self.max_bars:
            self.start += self.n - self.max_bars
            self.n = self.max_bars
        self.bars.append(bar)

    def arrays(self, upto: int) -> _Arrays:
        d = self.data[:, self.start : self.start + upto]
        return _Arrays(*(d[i] for i in range(len(_FIELDS) - 1)))

    def count_upto(self, now: int) -> int:
        ends = self.data[_FIELDS.index("end_ts"), self.start : self.start + self.n]
        return int(np.searchsorted(ends, now, side="right"))


class FeatureEngine:
    """Owns bar history per symbol and builds :class:`FeatureView` objects."""

    def __init__(self, symbols: tuple[str, ...] | list[str], max_bars: int = 720) -> None:
        self.max_bars = max_bars
        self._buf: dict[str, _SymbolBuffer] = {s: _SymbolBuffer(max_bars) for s in symbols}

    @property
    def history(self) -> dict[str, deque[Bar]]:
        return {s: b.bars for s, b in self._buf.items()}

    def add_bar(self, bar: Bar) -> None:
        buf = self._buf[bar.symbol]
        if buf.bars and bar.end_ts <= buf.bars[-1].end_ts:
            raise ValueError(f"non-monotonic bar for {bar.symbol}: {bar.end_ts} <= {buf.bars[-1].end_ts}")
        buf.append(bar)

    def view(self, symbol: str, now: int, signals: SignalBoard | None = None) -> FeatureView:
        """A view over bars with ``end_ts <= now`` only — later bars are unreachable."""
        buf = self._buf[symbol]
        k = buf.count_upto(now)
        bars = list(buf.bars)[:k] if k < len(buf.bars) else list(buf.bars)
        return FeatureView(symbol=symbol, now=now, bars=bars, signals=signals, _arr=buf.arrays(k))
