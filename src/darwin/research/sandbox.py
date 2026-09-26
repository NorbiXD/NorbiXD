"""Level-2 meta-research pipeline: proposal → sandbox → validation → promotion.

Every stage is recorded; a proposal is promoted only if all pass:

1. ``static``      DSL allowlist validation (:mod:`darwin.research.dsl`)
2. ``unit``        finite output in [-1, 1]; deterministic; order-independent (stateless);
                   not dead code; bounded latency — on real FeatureViews
3. ``leakage``     identical scores whether or not future bars exist in the engine
4. ``train``       historical replay with fees, latency, slippage, funding (frozen population of
                   random parameterisations of the new primitive)
5. ``holdout``     best train genome re-evaluated on a later, unseen window
6. ``challenger``  compared with the incumbent champion (or a reference genome) on identical
                   holdout bars

Stages 2-6 run in a separate Python process (:mod:`darwin.research.worker`) with a scrubbed
environment (no exchange/LLM credentials reachable), CPU/memory/file-size rlimits, no database
and a wall-clock timeout. The DSL makes in-process execution safe after promotion; the
subprocess additionally protects the research run itself.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from darwin.agents.genome import Genome
from darwin.agents.primitives import PRIMITIVES, Primitive, register_primitive
from darwin.core.ids import content_hash
from darwin.research.dsl import DSLError, compile_score, validate

SECRET_MARKERS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASS",
    "CREDENTIAL",
    "AUTH",
    "BYBIT",
    "XAI",
    "TYPESAFE",
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Proposal:
    source: str
    rationale: str = ""
    proposer: str = "unknown"

    @property
    def proposal_id(self) -> str:
        return "P" + content_hash(self.source, length=10)

    @property
    def primitive_name(self) -> str:
        return f"x_{self.proposal_id.lower()}"


@dataclass
class StageResult:
    stage: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxReport:
    proposal_id: str
    primitive: str
    passed: bool
    stages: list[StageResult]
    best_genome: dict[str, Any] | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxSettings:
    train_hours: float = 36.0
    holdout_hours: float = 12.0
    genomes: int = 6
    seed: int = 7
    min_holdout_trades: int = 5
    challenger_margin: float = 0.5
    timeout_s: float = 600.0
    cpu_seconds: int = 900
    memory_bytes: int = 3 * 1024**3


def scrubbed_env() -> dict[str, str]:
    """Environment for the sandbox process: only what Python needs, nothing secret-looking."""
    keep = {"PATH", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONHASHSEED", "TZ"}
    env = {
        k: v for k, v in os.environ.items() if k in keep and not any(m in k.upper() for m in SECRET_MARKERS)
    }
    src = str(Path(__file__).resolve().parents[2])  # .../src so `darwin` is importable
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HOME"] = tempfile.gettempdir()
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # RLIMIT_FSIZE=0: the child may not write any file
    return env


def _limits(settings: SandboxSettings) -> None:  # runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (settings.cpu_seconds, settings.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (settings.memory_bytes, settings.memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes of any size
    os.setsid()


def evaluate_proposal(
    proposal: Proposal, settings: SandboxSettings | None = None, champion: Genome | None = None
) -> SandboxReport:
    settings = settings or SandboxSettings()
    try:
        validate(proposal.source)
    except DSLError as e:  # cheap rejection before spending a process
        return SandboxReport(
            proposal.proposal_id,
            proposal.primitive_name,
            False,
            [StageResult("static", False, {"error": str(e)})],
        )
    payload = {
        "source": proposal.source,
        "primitive": proposal.primitive_name,
        "settings": asdict(settings),
        "champion": champion.model_dump(mode="json") if champion else None,
    }
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "darwin.research.worker"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=settings.timeout_s,
                env=scrubbed_env(),
                cwd=cwd,
                preexec_fn=lambda: _limits(settings),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxReport(
                proposal.proposal_id,
                proposal.primitive_name,
                False,
                [StageResult("static", True)],
                error="sandbox timeout",
            )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return SandboxReport(
            proposal.proposal_id,
            proposal.primitive_name,
            False,
            [StageResult("static", True)],
            error=f"worker exited {proc.returncode}: {proc.stderr[-2000:]}",
        )
    raw = json.loads(lines[-1])
    stages = [StageResult(s["stage"], s["passed"], s.get("detail", {})) for s in raw["stages"]]
    return SandboxReport(
        proposal_id=proposal.proposal_id,
        primitive=proposal.primitive_name,
        passed=all(s.passed for s in stages) and bool(stages),
        stages=stages,
        best_genome=raw.get("best_genome"),
        error=raw.get("error"),
    )


def make_primitive(name: str, source: str, origin: str) -> Primitive:
    parsed = validate(source)
    return Primitive(
        name=name,
        fn=compile_score(parsed),
        params=parsed.params,
        description=parsed.description or "sandbox species",
        origin=origin,
    )


class SpeciesRegistry:
    """Promoted species on disk. Loading re-validates the source (defence in depth)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def save(self, proposal: Proposal, report: SandboxReport) -> Path:
        if not report.passed:
            raise ValueError("only passing proposals can be promoted")
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{proposal.proposal_id}.json"
        path.write_text(
            json.dumps(
                {
                    "proposal_id": proposal.proposal_id,
                    "primitive": proposal.primitive_name,
                    "source": proposal.source,
                    "rationale": proposal.rationale,
                    "proposer": proposal.proposer,
                    "report": report.to_json(),
                },
                indent=2,
            )
        )
        return path

    def load(self, register: bool = True) -> list[tuple[str, Genome | None]]:
        """Register every promoted primitive; return (primitive, best genome) pairs."""
        out: list[tuple[str, Genome | None]] = []
        if not self.dir.exists():
            return out
        for path in sorted(self.dir.glob("P*.json")):
            rec = json.loads(path.read_text())
            name = rec["primitive"]
            report = rec.get("report") or {}
            stages = report.get("stages") or []
            if not report.get("passed") or not stages or not all(st.get("passed") for st in stages):
                log.warning("registry: %s has no passing sandbox report; not loaded", path.name)
                continue
            if register and name not in PRIMITIVES:
                register_primitive(make_primitive(name, rec["source"], f"sandbox:{rec['proposal_id']}"))
            best = rec["report"].get("best_genome")
            out.append((name, Genome.model_validate(best) if best else None))
        return out
