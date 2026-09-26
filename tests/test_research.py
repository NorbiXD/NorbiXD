from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from darwin.agents.primitives import PRIMITIVES
from darwin.core.types import LineageEventKind
from darwin.research.dsl import DSLError, compile_score, validate
from darwin.research.proposer import TEMPLATES, LLMProposer, TemplateProposer
from darwin.research.sandbox import (
    Proposal,
    SandboxReport,
    SandboxSettings,
    SpeciesRegistry,
    StageResult,
    evaluate_proposal,
    scrubbed_env,
)
from darwin.runtime.replay import build_replay
from tests.conftest import T0, make_config

GOOD = """DESCRIPTION = "toy"
PARAMS = {"lb": [10, 120, "int"], "s": [0.5, 3.0]}

def score(v, p):
    z = v.zret(int(p["lb"]))
    if not math.isfinite(z):
        return 0.0
    return math.tanh(z / p["s"])
"""


def _with_body(body: str) -> str:
    return f'PARAMS = {{"a": [1, 2]}}\n\ndef score(v, p):\n{body}\n'


@pytest.mark.parametrize(
    "body",
    [
        "    import os\n    return 0.0",
        "    return open('/etc/passwd')",
        "    return ().__class__.__bases__[0].__subclasses__()",
        "    return getattr(v, 'bars')",
        "    return v.bars[0].close",  # v.bars is not in the feature API
        "    return v.a.close.tofile('x')",  # numpy escape
        "    x = [i for i in range(3)]\n    return 0.0",
        "    for i in range(10):\n        pass\n    return 0.0",
        "    while True:\n        pass",
        "    f = lambda x: x\n    return 0.0",
        "    return 10 ** 10 ** 10",  # CPU/memory DoS
        "    return p['nope']",
        "    return max(*[1, 2])",
        "    return v.ret(n=5)",
        "    math = 1\n    return 0.0",
        "    _x = 1\n    return 0.0",
        "    return eval('1')",
        "    return math.os.system('id')",
        "    try:\n        return 0.0\n    except Exception:\n        return 0.0",
        # QM iteration 2: unbounded str/int values, annotations, round
        "    x = 'a' * 999999999\n    return 0.0",  # a gigabyte without a loop
        "    return len('abc')",
        "    return v.signal('x' * 40)",
        "    return v.signal('" + "t" * 33 + "')",
        "    return v.signal('a b')",
        "    return round(1.5)",
        "    return int(p['a']) * 1.0",  # int() only inside feature-call arguments
        "    return None",
    ],
)
def test_dsl_rejects_escapes(body: str) -> None:
    with pytest.raises(DSLError):
        validate(_with_body(body))


def test_dsl_rejects_annotations_evaluated_at_definition_time() -> None:
    for sig in ("def score(v: math.os, p):", "def score(v, p: open('x')):", "def score(v, p) -> 1:"):
        with pytest.raises(DSLError):
            validate(GOOD.replace("def score(v, p):", sig))


def test_dsl_values_are_bounded_floats_at_runtime() -> None:
    src = _with_body(
        "    big = 999999999 * 999999999 * 999999999 * 999999999\n"
        "    f = math.floor(big)\n"
        "    n = v.ret(3 * 2)\n"
        "    return f"
    )
    fn = compile_score(validate(src))

    class V:
        def ret(self, n: int) -> float:
            assert isinstance(n, int) and n == 6  # lookback math keeps ints
            return 0.0

    out = fn(V(), {"a": 1.5})
    assert isinstance(out, float)  # int constants compiled as floats; floor returns a float
    assert validate(_with_body("    return v.signal('x_narrative') + v.ret(int(p['a']) * 2)"))


def test_dsl_rejects_bad_structure() -> None:
    with pytest.raises(DSLError):
        validate("import math\n" + GOOD)
    with pytest.raises(DSLError):
        validate(GOOD.replace("def score(v, p):", "def score(v, p, q=1):"))
    with pytest.raises(DSLError):
        validate(GOOD.replace('"lb": [10, 120, "int"]', '"lb": [120, 10, "int"]'))
    with pytest.raises(DSLError):
        validate(GOOD + "\nx = 1\n")


def test_dsl_accepts_templates_and_compiles_into_empty_namespace() -> None:
    parsed = validate(GOOD)
    assert set(parsed.params) == {"lb", "s"} and parsed.params["lb"].kind == "int"
    fn = compile_score(parsed)
    assert "open" not in fn.__globals__["__builtins__"] and "__import__" not in fn.__globals__["__builtins__"]
    assert set(vars(fn.__globals__["math"])) <= {
        "tanh",
        "exp",
        "log",
        "sqrt",
        "copysign",
        "fabs",
        "isfinite",
        "floor",
        "ceil",
        "erf",
    }
    for _desc, src in TEMPLATES:
        validate(src)


def test_scrubbed_env_hides_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYBIT_API_SECRET", "s3cret")
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setenv("SOME_TOKEN", "t")
    env = scrubbed_env()
    assert not any(k in env for k in ("BYBIT_API_SECRET", "XAI_API_KEY", "SOME_TOKEN"))
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and "PYTHONPATH" in env


@pytest.mark.slow
def test_sandbox_pipeline_runs_isolated_and_reports_every_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYBIT_API_SECRET", "must-not-leak")
    prop = Proposal(source=GOOD, rationale="toy momentum", proposer="test")
    rep = evaluate_proposal(prop, SandboxSettings(train_hours=12, holdout_hours=6, genomes=3, timeout_s=300))
    assert rep.error is None, rep.error
    names = [s.stage for s in rep.stages]
    assert names == ["static", "unit", "leakage", "train", "holdout", "challenger"]
    static = rep.stages[0]
    assert static.passed and static.detail["env_secrets_visible"] == []
    assert rep.stages[1].passed and rep.stages[2].passed  # stateless, deterministic, no leakage
    assert rep.best_genome is not None and rep.best_genome["terms"][0]["primitive"] == prop.primitive_name
    assert rep.passed == all(s.passed for s in rep.stages)


def test_static_rejection_does_not_spawn_a_process() -> None:
    rep = evaluate_proposal(Proposal(source=_with_body("    return open('x')")))
    assert not rep.passed and [s.stage for s in rep.stages] == ["static"]


def test_registry_promotion_registers_primitive_and_injects_challenger(tmp_path: Path) -> None:
    prop = Proposal(source=GOOD, rationale="toy", proposer="test")
    from darwin.agents.genome import GeneTerm, Genome, RiskGenes

    genome = Genome(
        terms=(GeneTerm(primitive="momentum", params={"lookback": 30, "scale": 1.0}, weight=1.0),),
        symbols=("BTCUSDT",),
        entry_threshold=0.5,
        exit_threshold=0.0,
        risk=RiskGenes(exposure=1.0, stop_loss_pct=0.02, take_profit_pct=0.05, max_hold_bars=60),
    )
    genome = genome.model_copy(
        update={
            "terms": (
                GeneTerm.model_construct(
                    primitive=prop.primitive_name, params={"lb": 30.0, "s": 1.0}, weight=1.0
                ),
            )
        }
    )
    rep = SandboxReport(
        prop.proposal_id,
        prop.primitive_name,
        True,
        [StageResult(s, True) for s in ("static", "unit", "leakage", "train", "holdout", "challenger")],
        best_genome=genome.model_dump(mode="json"),
    )
    reg = SpeciesRegistry(tmp_path / "species")
    with pytest.raises(ValueError):
        reg.save(prop, SandboxReport(prop.proposal_id, prop.primitive_name, False, []))
    reg.save(prop, rep)
    loaded = reg.load(register=True)
    assert loaded[0][0] == prop.primitive_name and prop.primitive_name in PRIMITIVES
    assert PRIMITIVES[prop.primitive_name].origin == f"sandbox:{prop.proposal_id}"
    h = build_replay(make_config(), iter([]), T0)
    g = loaded[0][1]
    assert g is not None
    a = h.engine.inject_genome(g, origin=f"sandbox:{prop.primitive_name}")
    assert a.alive and any(r["kind"] == LineageEventKind.PROPOSED.value for r in h.engine.recent_lineage)


def test_proposers() -> None:
    props = TemplateProposer(seed=1).propose(2)
    assert len(props) == 2 and all(validate(p.source) for p in props)

    async def fake_llm(prompt: str) -> str:
        assert "v.<feature>" in prompt and "zret" in prompt
        return "Idea: toy momentum\n```python\n" + GOOD + "```\nIdea: broken\n```python\nimport os\n```"

    got = asyncio.run(
        LLMProposer(fake_llm).propose(
            3,
            [
                {
                    "species": "momentum",
                    "regime": "trend_up",
                    "mean_return": 0.002,
                    "t_stat": 2.5,
                    "trades": 40,
                }
            ],
        )
    )
    assert len(got) == 2 and got[0].rationale == "Idea: toy momentum"
    # the LLM's output is only a proposal: the broken one fails static validation
    with pytest.raises(DSLError):
        validate(got[1].source)
    assert os.environ.get("DARWIN_NEVER_SET") is None


_SQUARE_40 = "".join("    x = x * x\n" for _ in range(40))


@pytest.mark.parametrize(
    "prelude",
    [
        "    x = True + True\n",  # QM iteration 3, M4: bools are ints
        "    x = True\n    x += True\n",
        "    x = (v.ret(1) > 0) + (v.ret(2) > 0) + 2\n",  # comparison results are bools
        "    x = -True + -True\n",
        "    x = abs(True) + max(True, False) + 1\n",
        "    x = 3\n    if v.funding(1) > 0.01:\n        x = x + 1\n",  # a bomb behind a rare trigger
    ],
)
def test_dsl_arithmetic_is_float_only_so_squaring_bombs_overflow_instead_of_allocating(prelude: str) -> None:
    import time

    src = _with_body(prelude + _SQUARE_40 + "    return x")
    parsed = validate(src)  # it is valid DSL...
    fn = compile_score(parsed)

    class V:
        def ret(self, n: int) -> float:
            return 0.01

        def funding(self, n: int) -> float:
            return 0.05  # trigger fires

    t0 = time.perf_counter()
    out = fn(V(), {"a": 1.5})
    assert time.perf_counter() - t0 < 0.05  # ...but it cannot build a 2**40-bit integer
    assert isinstance(out, float) and out == float("inf")


def test_dsl_rejects_augmented_assignment_to_non_names() -> None:
    with pytest.raises(DSLError):
        validate(_with_body("    p['a'] += 1\n    return 0.0"))


def test_slow_species_are_quarantined_by_the_decision_time_budget() -> None:
    import time

    from darwin.agents.genome import GeneTerm, Genome, RiskGenes
    from darwin.agents.params import ParamSpec
    from darwin.agents.primitives import Primitive, register_primitive
    from darwin.market.synthetic import SyntheticMarket

    def slow(v: object, p: object) -> float:
        time.sleep(0.03)
        return 0.0

    register_primitive(Primitive("test_slow", slow, {"k": ParamSpec(1.0, 2.0)}, "slow"), replace=True)
    try:
        cfg = make_config(challenge={"duration_hours": 4}, evolution={"max_decide_ms": 10})
        g = Genome(
            terms=(GeneTerm(primitive="test_slow", params={"k": 1.5}, weight=1.0),),
            symbols=("BTCUSDT",),
            entry_threshold=0.3,
            exit_threshold=0.0,
            risk=RiskGenes(exposure=1.0, stop_loss_pct=0.02, take_profit_pct=0.05, max_hold_bars=60),
        )
        m = SyntheticMarket(("BTCUSDT", "ETHUSDT"), T0, 4 * 3_600_000 + 60_000, step_ms=5_000, seed=2)
        h = build_replay(cfg, m.events(), T0, seed_genomes=[(g, "sandbox:test")])
        (agent,) = [a for a in h.engine.population.agents.values() if a.genome.genome_id == g.genome_id]
        h.driver.run()
        assert not agent.alive and agent.death_reason == "runtime_error"
        assert h.engine.stats["quarantined"] == 1 and h.engine.ended
    finally:
        PRIMITIVES.pop("test_slow", None)
