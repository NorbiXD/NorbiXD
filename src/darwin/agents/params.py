"""Parameter search-space specification shared by primitives and genomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    low: float
    high: float
    kind: Literal["float", "int", "choice"] = "float"
    log: bool = False
    choices: tuple[str, ...] = ()
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("choice param needs choices")
        elif self.high <= self.low:
            raise ValueError("ParamSpec high must be > low")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled ParamSpec needs low > 0")

    # Normalised coordinates in [0, 1] make mutation scale-free across parameters.
    def to_unit(self, value: float) -> float:
        if self.kind == "choice":
            n = len(self.choices)
            return (int(value) + 0.5) / n
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        if self.kind == "choice":
            return float(min(int(u * len(self.choices)), len(self.choices) - 1))
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        if self.kind == "int":
            return float(round(v))
        return float(v)

    def sample(self, rng: np.random.Generator) -> float:
        return self.from_unit(float(rng.random()))

    def clip(self, value: float) -> float:
        if self.kind == "choice":
            return float(min(max(int(value), 0), len(self.choices) - 1))
        v = min(max(value, self.low), self.high)
        return float(round(v)) if self.kind == "int" else float(v)

    def mutate(self, value: float, sigma: float, rng: np.random.Generator) -> float:
        if self.kind == "choice":
            if rng.random() < sigma * 2:
                return float(rng.integers(0, len(self.choices)))
            return value
        u = self.to_unit(value) + float(rng.normal(0.0, sigma))
        # reflect at the boundaries so mass does not pile up on the edges
        if u < 0:
            u = -u
        if u > 1:
            u = 2 - u
        return self.from_unit(u)
