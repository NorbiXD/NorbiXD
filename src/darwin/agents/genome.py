"""Serializable agent genomes, plus mutation and crossover operators.

A genome is a small program in a strategy DSL:

    score  = Σ_i weight_i · primitive_i(features; params_i) / Σ_i |weight_i|
    gate   = current regime ∈ allowed_regimes   (empty set = all regimes)
    flat  -> enter long/short when |score| >= entry_threshold
    long  -> exit when score < exit_threshold (flip if score <= -entry_threshold)
    size  -> target_exposure = sign · exposure · (|score| if confidence_scaling else 1)

Risk genes are *requests*: the Risk Governor decides the final quantity. Everything here is
data — genomes are persisted verbatim and their content hash is the genome id.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from darwin.agents.params import ParamSpec
from darwin.agents.primitives import PRIMITIVES
from darwin.core.ids import content_hash
from darwin.core.types import Regime, Urgency

TRADEABLE_REGIMES = (Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE, Regime.HIGH_VOL)

# Scalar genes and their search spaces (the per-primitive spaces live on the primitives).
SCALAR_SPACE: dict[str, ParamSpec] = {
    "entry_threshold": ParamSpec(0.15, 0.95),
    "exit_threshold": ParamSpec(-0.6, 0.6),
    "cooldown_bars": ParamSpec(0, 30, "int"),
    "risk.exposure": ParamSpec(0.25, 5.0, log=True, doc="requested notional / capital"),
    "risk.stop_loss_pct": ParamSpec(0.003, 0.08, log=True),
    "risk.take_profit_pct": ParamSpec(0.004, 0.25, log=True),
    "risk.max_hold_bars": ParamSpec(5, 1440, "int", log=True),
    "execution.max_slippage_bps": ParamSpec(3.0, 40.0, log=True),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GeneTerm(_Frozen):
    primitive: str
    params: dict[str, float]
    weight: float = Field(ge=-1.0, le=1.0)

    @field_validator("primitive")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in PRIMITIVES:
            raise ValueError(f"unknown primitive {v!r}")
        return v


class RiskGenes(_Frozen):
    exposure: float = Field(gt=0)
    stop_loss_pct: float = Field(gt=0)
    take_profit_pct: float = Field(gt=0)
    max_hold_bars: int = Field(ge=1)
    confidence_scaling: bool = True


class ExecutionGenes(_Frozen):
    urgency: Urgency = Urgency.MARKET
    max_slippage_bps: float = Field(default=15.0, gt=0)


class Genome(_Frozen):
    terms: tuple[GeneTerm, ...] = Field(min_length=1)
    symbols: tuple[str, ...] = Field(min_length=1)
    entry_threshold: float
    exit_threshold: float
    cooldown_bars: int = 0
    allowed_regimes: tuple[Regime, ...] = ()
    risk: RiskGenes
    execution: ExecutionGenes = ExecutionGenes()
    provider: str | None = None  # model provider consulted by model-backed primitives
    prompt_version: str | None = None

    @property
    def genome_id(self) -> str:
        return "G" + content_hash(self.model_dump(mode="json"), length=11)

    @property
    def species(self) -> str:
        ranked = sorted(self.terms, key=lambda t: -abs(t.weight))
        names = []
        for t in ranked:
            n = t.primitive if t.weight >= 0 else f"anti-{t.primitive}"
            if n not in names:
                names.append(n)
        if len(names) == 1:
            return names[0]
        return "hybrid:" + "+".join(names)

    @property
    def lookback_bars(self) -> int:
        """Rough warm-up requirement: the largest bar-count parameter in the genome."""
        m = 62
        for t in self.terms:
            spec = PRIMITIVES[t.primitive].params
            for k, v in t.params.items():
                if spec[k].kind == "int":
                    m = max(m, int(v) + 2)
        return m

    def scalar(self, path: str) -> float:
        obj: Any = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return float(obj)

    def flat(self) -> dict[str, float]:
        """Flattened numeric view (used for distance, mutation diffs and lineage display)."""
        out = {p: self.scalar(p) for p in SCALAR_SPACE}
        for i, t in enumerate(self.terms):
            out[f"terms[{i}].{t.primitive}.weight"] = t.weight
            for k, v in t.params.items():
                out[f"terms[{i}].{t.primitive}.{k}"] = v
        return out


# ----------------------------------------------------------------------------- construction


def random_term(primitive: str, rng: np.random.Generator) -> GeneTerm:
    spec = PRIMITIVES[primitive]
    params = {k: s.sample(rng) for k, s in spec.params.items()}
    return GeneTerm(primitive=primitive, params=params, weight=float(rng.uniform(0.4, 1.0)))


def random_genome(
    rng: np.random.Generator,
    symbols: tuple[str, ...],
    primitive: str | None = None,
    max_terms: int = 1,
) -> Genome:
    names = sorted(PRIMITIVES)
    first = primitive or names[int(rng.integers(0, len(names)))]
    terms = [random_term(first, rng)]
    n_extra = int(rng.integers(0, max_terms)) if max_terms > 1 else 0
    for _ in range(n_extra):
        terms.append(random_term(names[int(rng.integers(0, len(names)))], rng))
    n_sym = int(rng.integers(1, len(symbols) + 1))
    chosen = tuple(sorted(rng.choice(list(symbols), size=n_sym, replace=False).tolist()))
    sp = SCALAR_SPACE
    return Genome(
        terms=tuple(terms),
        symbols=chosen,
        entry_threshold=sp["entry_threshold"].sample(rng),
        exit_threshold=sp["exit_threshold"].sample(rng),
        cooldown_bars=int(sp["cooldown_bars"].sample(rng)),
        allowed_regimes=(),
        risk=RiskGenes(
            exposure=sp["risk.exposure"].sample(rng),
            stop_loss_pct=sp["risk.stop_loss_pct"].sample(rng),
            take_profit_pct=sp["risk.take_profit_pct"].sample(rng),
            max_hold_bars=int(sp["risk.max_hold_bars"].sample(rng)),
            confidence_scaling=bool(rng.random() < 0.5),
        ),
        execution=ExecutionGenes(max_slippage_bps=sp["execution.max_slippage_bps"].sample(rng)),
    )


# ----------------------------------------------------------------------------- variation


class Mutation(_Frozen):
    """One recorded change, e.g. ``terms[0].momentum.lookback: 20 -> 34``."""

    path: str
    before: Any
    after: Any


def _set_scalars(g: Genome, values: dict[str, float]) -> Genome:
    risk = g.risk.model_copy(
        update={
            "exposure": values["risk.exposure"],
            "stop_loss_pct": values["risk.stop_loss_pct"],
            "take_profit_pct": values["risk.take_profit_pct"],
            "max_hold_bars": int(values["risk.max_hold_bars"]),
        }
    )
    execution = g.execution.model_copy(update={"max_slippage_bps": values["execution.max_slippage_bps"]})
    return g.model_copy(
        update={
            "entry_threshold": values["entry_threshold"],
            "exit_threshold": values["exit_threshold"],
            "cooldown_bars": int(values["cooldown_bars"]),
            "risk": risk,
            "execution": execution,
        }
    )


def _revalidate(g: Genome) -> Genome:
    # model_copy skips validation; round-trip to guarantee every mutated genome is valid.
    return Genome.model_validate(g.model_dump())


def mutate(
    g: Genome,
    rng: np.random.Generator,
    sigma: float,
    universe: tuple[str, ...],
    structural_rate: float = 0.15,
    max_terms: int = 3,
) -> tuple[Genome, list[Mutation]]:
    changes: list[Mutation] = []
    # scalar genes: each mutates with probability 0.5
    scal = {p: g.scalar(p) for p in SCALAR_SPACE}
    for path, spec in SCALAR_SPACE.items():
        if rng.random() < 0.5:
            new = spec.mutate(scal[path], sigma, rng)
            if new != scal[path]:
                changes.append(Mutation(path=path, before=round(scal[path], 6), after=round(new, 6)))
                scal[path] = new
    out = _set_scalars(g, scal)

    # term parameters and weights
    terms = []
    for i, t in enumerate(out.terms):
        pspace = PRIMITIVES[t.primitive].params
        params = dict(t.params)
        for k, s in pspace.items():
            if rng.random() < 0.5:
                new = s.mutate(params[k], sigma, rng)
                if new != params[k]:
                    changes.append(
                        Mutation(path=f"terms[{i}].{t.primitive}.{k}", before=params[k], after=new)
                    )
                    params[k] = new
        w = t.weight
        if rng.random() < 0.5:
            nw = float(np.clip(w + rng.normal(0, sigma), -1.0, 1.0))
            if abs(nw) < 0.05:
                nw = 0.05 if w >= 0 else -0.05
            changes.append(
                Mutation(path=f"terms[{i}].{t.primitive}.weight", before=round(w, 4), after=round(nw, 4))
            )
            w = nw
        terms.append(GeneTerm(primitive=t.primitive, params=params, weight=w))

    # structural mutations: add / drop / swap a term, toggle a regime, change universe
    if rng.random() < structural_rate:
        op = rng.choice(["add", "drop", "regime", "symbol", "invert", "scaling"])
        names = sorted(PRIMITIVES)
        if op == "add" and len(terms) < max_terms:
            prim = names[int(rng.integers(0, len(names)))]
            terms.append(random_term(prim, rng))
            changes.append(Mutation(path="terms", before=None, after=f"+{prim}"))
        elif op == "drop" and len(terms) > 1:
            j = int(rng.integers(0, len(terms)))
            dropped = terms.pop(j)
            changes.append(Mutation(path="terms", before=dropped.primitive, after=None))
        elif op == "regime":
            reg = TRADEABLE_REGIMES[int(rng.integers(0, len(TRADEABLE_REGIMES)))]
            allowed = set(out.allowed_regimes) or set(TRADEABLE_REGIMES)
            before = sorted(r.value for r in out.allowed_regimes)
            allowed ^= {reg}
            if not allowed or allowed == set(TRADEABLE_REGIMES):
                allowed = set()
            out = out.model_copy(update={"allowed_regimes": tuple(sorted(allowed, key=lambda r: r.value))})
            changes.append(
                Mutation(path="allowed_regimes", before=before, after=sorted(r.value for r in allowed))
            )
        elif op == "symbol" and len(universe) > 1:
            sym = universe[int(rng.integers(0, len(universe)))]
            cur = set(out.symbols)
            nxt = cur ^ {sym}
            if nxt:
                out = out.model_copy(update={"symbols": tuple(sorted(nxt))})
                changes.append(Mutation(path="symbols", before=sorted(cur), after=sorted(nxt)))
        elif op == "invert":
            j = int(rng.integers(0, len(terms)))
            t = terms[j]
            terms[j] = GeneTerm(primitive=t.primitive, params=t.params, weight=-t.weight)
            changes.append(
                Mutation(path=f"terms[{j}].{t.primitive}.weight", before=t.weight, after=-t.weight)
            )
        elif op == "scaling":
            flag = not out.risk.confidence_scaling
            out = out.model_copy(update={"risk": out.risk.model_copy(update={"confidence_scaling": flag})})
            changes.append(Mutation(path="risk.confidence_scaling", before=not flag, after=flag))
    out = out.model_copy(update={"terms": tuple(terms)})
    return _revalidate(out), changes


def crossover(a: Genome, b: Genome, rng: np.random.Generator, max_terms: int = 3) -> Genome:
    """Uniform crossover. Same-structure parents mix parameters; different ones mix terms."""
    if [t.primitive for t in a.terms] == [t.primitive for t in b.terms]:
        terms = []
        for ta, tb in zip(a.terms, b.terms, strict=True):
            params = {k: (ta.params[k] if rng.random() < 0.5 else tb.params[k]) for k in ta.params}
            weight = ta.weight if rng.random() < 0.5 else tb.weight
            terms.append(GeneTerm(primitive=ta.primitive, params=params, weight=weight))
    else:
        pool = list(a.terms) + list(b.terms)
        k = int(rng.integers(2, min(max_terms, len(pool)) + 1)) if len(pool) >= 2 else 1
        idx = rng.choice(len(pool), size=k, replace=False)
        # always keep at least one term from each parent so the child is a genuine hybrid
        chosen = [pool[int(i)] for i in sorted(idx)]
        if all(t in a.terms for t in chosen):
            chosen[-1] = b.terms[0]
        elif all(t in b.terms for t in chosen):
            chosen[-1] = a.terms[0]
        terms = chosen
    scal_a = {p: a.scalar(p) for p in SCALAR_SPACE}
    scal_b = {p: b.scalar(p) for p in SCALAR_SPACE}
    mixed = {p: (scal_a[p] if rng.random() < 0.5 else scal_b[p]) for p in SCALAR_SPACE}
    base = a if rng.random() < 0.5 else b
    child = _set_scalars(base, mixed)
    symbols = tuple(sorted(set(a.symbols) | set(b.symbols))) if rng.random() < 0.3 else base.symbols
    regimes = a.allowed_regimes if rng.random() < 0.5 else b.allowed_regimes
    child = child.model_copy(update={"terms": tuple(terms), "symbols": symbols, "allowed_regimes": regimes})
    return _revalidate(child)


def genome_distance(a: Genome, b: Genome) -> float:
    """Normalised distance in [0, 1]; 1.0 for structurally different genomes."""
    if [t.primitive for t in a.terms] != [t.primitive for t in b.terms] or a.symbols != b.symbols:
        return 1.0
    diffs = []
    for p, spec in SCALAR_SPACE.items():
        diffs.append(abs(spec.to_unit(a.scalar(p)) - spec.to_unit(b.scalar(p))))
    for ta, tb in zip(a.terms, b.terms, strict=True):
        spec_p = PRIMITIVES[ta.primitive].params
        for k, s in spec_p.items():
            diffs.append(abs(s.to_unit(ta.params[k]) - s.to_unit(tb.params[k])))
        diffs.append(abs(ta.weight - tb.weight) / 2)
    return float(np.mean(diffs)) if diffs else 0.0
