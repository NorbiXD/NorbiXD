"""The Level-2 species DSL: a statically-verified subset of Python for generated strategies.

A proposed species is source code of this exact shape::

    DESCRIPTION = "fade funding extremes, but only while volatility is expanding"
    PARAMS = {"fw": [30, 720, "int"], "z": [0.5, 3.0], "short": [5, 60, "int"], "long": [60, 480, "int"]}

    def score(v, p):
        fz = v.funding_z(int(p["fw"]))
        vr = v.vol_ratio(int(p["short"]), int(p["long"]))
        if not (math.isfinite(fz) and math.isfinite(vr)):
            return 0.0
        if vr < 1.2 or abs(fz) < p["z"]:
            return 0.0
        return -math.tanh(fz / p["z"])

Why a DSL and not arbitrary Python: a promoted species runs inside the trading process, so the
*language itself* must make escape impossible, not just a runtime sandbox. The validator is an
allowlist over the AST:

* no imports, loops, comprehensions, lambdas, try/with, globals, nested defs, star-args, keywords,
  decorators or annotations (annotations are evaluated at definition time);
* the only attribute access permitted is ``v.<feature API method>`` and ``math.<function>`` — so
  no ``__class__``/``__subclasses__`` escapes, no numpy methods like ``ndarray.tofile``;
* calls only to those attributes or to ``abs/min/max/float``; ``int(...)`` only inside the
  arguments of a feature call (lookbacks);
* subscripts only as ``p["<declared param>"]``; no names starting with ``_``;
* **bounded values, bounded time.** No ``**`` (``10 ** 10 ** 10`` needs no loop), no string
  constants except parameter keys and ``v.signal("<topic>")`` (``"a" * 999999999`` would allocate
  a gigabyte). Outside feature arguments every arithmetic operand is coerced with ``float()`` at
  compile time — constants, names, comparison results and bools alike — so repeated squaring
  (``x = True + True`` then ``x = x * x`` forty times would be a 2^40-bit integer) overflows to
  ``inf`` instead of allocating memory; ``math.floor/ceil`` return floats. Inside feature
  arguments (lookbacks) integers stay integers, but no reassignment can happen inside a single
  expression, and every argument passes through a bounded-lookback check (a finite number,
  clamped to 0..10 000) before any feature sees it. ``p`` may only appear as ``p["key"]`` and
  ``v``/``math`` only as attribute owners, so no string can reach arithmetic (``min(p)`` would
  have returned a key). With a bounded node count, execution
  time is bounded too; the engine additionally quarantines any agent whose decision step
  exceeds a time budget. The compiled function runs in a namespace containing
  nothing else and is compiled without inheriting the host's ``__future__`` flags.

``FeatureView`` only exposes past bars, so the feature API is also the leakage boundary.
"""

from __future__ import annotations

import ast
import math
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from darwin.agents.params import ParamSpec

FEATURE_API = frozenset(
    {
        "close",
        "ret",
        "vol",
        "zret",
        "zprice",
        "rsi",
        "channel_position",
        "atr_pct",
        "vol_ratio",
        "flow_imbalance",
        "book_imbalance",
        "spread_bps",
        "funding",
        "funding_z",
        "oi_change",
        "liq_imbalance",
        "liq_intensity",
        "signal",
        "ready",
    }
)
MATH_API = frozenset({"tanh", "exp", "log", "sqrt", "copysign", "fabs", "isfinite", "floor", "ceil", "erf"})
BUILTINS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,  # validated: only inside feature-call arguments
}
RESERVED = frozenset({"v", "p", "math", *BUILTINS})
MAX_SOURCE_CHARS = 4_000
MAX_FUNCTION_NODES = 400
MAX_TOPIC_CHARS = 32

_ALLOWED_BODY = (
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Attribute,
    ast.Load,
    ast.Store,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Pass,
)


class DSLError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedSpecies:
    description: str
    params: dict[str, ParamSpec]
    source: str


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError as e:
        raise DSLError(f"line {node.lineno}: expected a literal") from e


def _parse_params(node: ast.expr) -> dict[str, ParamSpec]:
    raw = _literal(node)
    if not isinstance(raw, dict) or not raw or len(raw) > 8:
        raise DSLError("PARAMS must be a dict of 1..8 entries")
    out: dict[str, ParamSpec] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise DSLError(f"bad param name {name!r}")
        if not isinstance(spec, (list, tuple)) or len(spec) not in (2, 3):
            raise DSLError(f"param {name}: expected [low, high] or [low, high, kind]")
        low, high = float(spec[0]), float(spec[1])
        kind = spec[2] if len(spec) == 3 else "float"
        if kind not in ("float", "int", "log"):
            raise DSLError(f"param {name}: kind must be float|int|log")
        if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
            raise DSLError(f"param {name}: need finite low < high")
        if kind == "log":
            out[name] = ParamSpec(low, high, "float", log=True)
        elif kind == "int":
            out[name] = ParamSpec(low, high, "int", log=low > 0 and high / max(low, 1e-9) > 20)
        else:
            out[name] = ParamSpec(low, high)
    return out


def validate(source: str) -> ParsedSpecies:
    """Statically verify a proposal; raises :class:`DSLError` with the first violation."""
    if len(source) > MAX_SOURCE_CHARS:
        raise DSLError("source too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise DSLError(f"syntax error: {e}") from e
    description = ""
    params: dict[str, ParamSpec] | None = None
    fn: ast.FunctionDef | None = None
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue  # docstring
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            if target == "DESCRIPTION":
                d = _literal(stmt.value)
                if not isinstance(d, str):
                    raise DSLError("DESCRIPTION must be a string")
                description = d[:500]
                continue
            if target == "PARAMS":
                params = _parse_params(stmt.value)
                continue
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "score" and fn is None:
            fn = stmt
            continue
        raise DSLError(
            f"line {getattr(stmt, 'lineno', '?')}: only DESCRIPTION, PARAMS and def score(v, p) allowed"
        )
    if fn is None or params is None:
        raise DSLError("need PARAMS and def score(v, p)")
    a = fn.args
    if (
        [x.arg for x in a.args] != ["v", "p"]
        or any(x.annotation is not None or x.type_comment for x in a.args)
        or a.vararg
        or a.kwarg
        or a.kwonlyargs
        or a.defaults
        or a.kw_defaults
        or a.posonlyargs
        or fn.decorator_list
        or fn.returns
        or fn.type_comment
        or getattr(fn, "type_params", None)
    ):
        raise DSLError("signature must be exactly: def score(v, p)")
    nodes = list(ast.walk(ast.Module(body=fn.body, type_ignores=[])))
    if len(nodes) > MAX_FUNCTION_NODES:
        raise DSLError("function too large")
    in_feature_args = _feature_arg_nodes(fn)
    topic_args = {
        id(n.args[0]) for n in nodes if isinstance(n, ast.Call) and _is_feature_call(n, "signal") and n.args
    }
    param_keys = {id(n.slice) for n in nodes if isinstance(n, ast.Subscript)}
    # `p` only as p["key"], `v`/`math` only as attribute owners: a bare `p` would let
    # `min(p)` return a parameter *key* (a string) — QM iteration 4, M-B
    owners = {id(n.value) for n in nodes if isinstance(n, (ast.Subscript, ast.Attribute))}
    for node in nodes:
        if isinstance(node, ast.Name) and node.id in ("p", "v", "math") and id(node) not in owners:
            raise DSLError(f"line {node.lineno}: {node.id!r} may only be used as p[...] / v.<f> / math.<f>")
    for node in nodes:
        if isinstance(node, ast.Module):
            continue
        if not isinstance(node, _ALLOWED_BODY):
            raise DSLError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} not allowed")
        if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
            raise DSLError(f"line {node.lineno}: augmented assignment only to a plain name")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise DSLError(f"name {node.id!r} not allowed")
            if isinstance(node.ctx, ast.Store) and node.id in RESERVED:
                raise DSLError(f"cannot assign to {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or isinstance(node.ctx, ast.Store):
                raise DSLError(f"line {node.lineno}: attribute access only as v.<feature> or math.<fn>")
            owner, attr = node.value.id, node.attr
            if not ((owner == "v" and attr in FEATURE_API) or (owner == "math" and attr in MATH_API)):
                raise DSLError(f"line {node.lineno}: {owner}.{attr} is not in the allowed API")
        elif isinstance(node, ast.Call):
            if node.keywords:
                raise DSLError(f"line {node.lineno}: keyword arguments not allowed")
            f = node.func
            if isinstance(f, ast.Name):
                if f.id not in BUILTINS:
                    raise DSLError(f"line {node.lineno}: call to {f.id!r} not allowed")
                if f.id == "int" and id(node) not in in_feature_args:
                    raise DSLError(f"line {node.lineno}: int(...) only inside feature-call arguments")
            elif not isinstance(f, ast.Attribute):
                raise DSLError(f"line {node.lineno}: only v.<feature>(...), math.<fn>(...) or builtins")
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                raise DSLError("star-args not allowed")
        elif isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "p"):
                raise DSLError(f'line {node.lineno}: subscripts only as p["name"]')
            sl = node.slice
            if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str) and sl.value in params):
                raise DSLError(f"line {node.lineno}: unknown parameter subscript")
            if isinstance(node.ctx, ast.Store):
                raise DSLError("cannot assign into p")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool, str)):
                raise DSLError("unsupported constant")
            if isinstance(node.value, str):
                if id(node) in param_keys:
                    pass  # checked with the subscript
                elif id(node) in topic_args:
                    if len(node.value) > MAX_TOPIC_CHARS or not node.value.replace("_", "").isalnum():
                        raise DSLError(f"line {node.lineno}: bad signal topic")
                elif node.value:
                    raise DSLError(f"line {node.lineno}: string constants only as p[...] keys or topics")
                else:
                    raise DSLError(f"line {node.lineno}: empty string constant")
            if (
                isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
                and abs(node.value) > 1e9
            ):
                raise DSLError("numeric constant too large")
    return ParsedSpecies(description=description, params=params, source=source)


def _is_feature_call(node: ast.Call, name: str | None = None) -> bool:
    f = node.func
    return (
        isinstance(f, ast.Attribute)
        and isinstance(f.value, ast.Name)
        and f.value.id == "v"
        and (name is None or f.attr == name)
    )


def _feature_arg_nodes(fn: ast.FunctionDef) -> set[int]:
    """ids of every node inside the arguments of a ``v.<feature>(...)`` call (lookback math)."""
    out: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _is_feature_call(node):
            for arg in node.args:
                out.update(id(n) for n in ast.walk(arg))
    return out


def _as_float(node: ast.expr) -> ast.expr:
    return ast.copy_location(ast.Call(func=ast.Name("float", ast.Load()), args=[node], keywords=[]), node)


MAX_LOOKBACK = 10_000


def _lookback(x: object) -> int:
    """Every feature-call argument passes through here: a finite real number, clamped to a
    bounded integer window — whatever arithmetic produced it, a feature can never receive a
    string, a huge integer or a negative lookback."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"feature argument must be a number, got {type(x).__name__}")
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError("feature argument must be finite")
    return max(0, min(int(x), MAX_LOOKBACK))


class _BoundFeatureArgs(ast.NodeTransformer):
    """Wrap the arguments of ``v.<feature>(...)`` in ``__lookback__(...)`` (the name is not
    writable from source: names starting with ``_`` are rejected by the validator)."""

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if _is_feature_call(node) and not _is_feature_call(node, "signal"):
            node.args = [
                ast.copy_location(
                    ast.Call(func=ast.Name("__lookback__", ast.Load()), args=[a], keywords=[]), a
                )
                for a in node.args
            ]
        return node


class _FloatConstants(ast.NodeTransformer):
    """Outside feature-call arguments: integer constants become floats and every arithmetic
    operand is wrapped in ``float()``, so no body value can grow without bound (bools and
    comparison results included). Feature arguments keep integer semantics for lookbacks."""

    def __init__(self, keep: set[int]) -> None:
        self.keep = keep

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        v = node.value
        if isinstance(v, int) and not isinstance(v, bool) and id(node) not in self.keep:
            return ast.copy_location(ast.Constant(float(v)), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        keep = id(node) in self.keep
        self.generic_visit(node)
        if not keep:
            node.left, node.right = _as_float(node.left), _as_float(node.right)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        keep = id(node) in self.keep
        self.generic_visit(node)
        if not keep and isinstance(node.op, (ast.USub, ast.UAdd)):
            node.operand = _as_float(node.operand)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        # x op= y  ->  x = float(x) op float(y)   (targets are plain names: validated)
        self.generic_visit(node)
        assert isinstance(node.target, ast.Name)
        load = ast.copy_location(ast.Name(node.target.id, ast.Load()), node.target)
        value = ast.BinOp(left=_as_float(load), op=node.op, right=_as_float(node.value))
        assign = ast.Assign(targets=[node.target], value=ast.copy_location(value, node))
        return ast.copy_location(assign, node)


def _float_floor(x: float) -> float:
    return float(math.floor(x))


def _float_ceil(x: float) -> float:
    return float(math.ceil(x))


def compile_score(parsed: ParsedSpecies) -> Callable[[Any, Mapping[str, float]], float]:
    """Compile a *validated* proposal into ``score(v, p)`` with an empty-by-default namespace."""
    validate(parsed.source)  # never compile anything that has not passed the validator
    safe = {name: getattr(math, name) for name in MATH_API}
    safe.update(floor=_float_floor, ceil=_float_ceil)
    safe_math = types.SimpleNamespace(**safe)
    namespace: dict[str, Any] = {
        "__builtins__": dict(BUILTINS),
        "math": safe_math,
        "__lookback__": _lookback,
    }
    tree = ast.parse(parsed.source)
    fdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    tree = _FloatConstants(_feature_arg_nodes(fdef)).visit(tree)
    tree = ast.fix_missing_locations(_BoundFeatureArgs().visit(tree))
    code = compile(tree, "<species>", "exec", dont_inherit=True)
    exec(code, namespace)
    fn = namespace["score"]
    if not callable(fn):
        raise DSLError("score is not callable")
    return fn  # type: ignore[no-any-return]
