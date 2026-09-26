"""Account preflight for testnet/live: refuse to trade on an account we don't fully understand.

Checks (live: every failure is fatal; testnet: soft failures are reported as warnings):

1. wallet equity covers the challenge's starting capital (the ledger trades that capital only);
2. the account is **flat** on every challenge symbol — there is no crash-resume, so adopting
   unknown positions would break attribution and reconciliation;
3. margin mode is cross (``REGULAR_MARGIN``): the governor's liquidation model assumes it;
4. position mode is one-way (the engine sends ``positionIdx=0`` and nets agents);
5. per-symbol leverage is set so the exchange's initial margin never binds before our own limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from darwin.config.challenge import ChallengeConfig, InstrumentSpec
from darwin.exchange.bybit.rest import BybitRest


class PreflightError(RuntimeError):
    pass


@dataclass
class PreflightReport:
    equity: float = 0.0
    margin_mode: str = ""
    leverage: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


async def account_preflight(
    rest: BybitRest, cfg: ChallengeConfig, instruments: dict[str, InstrumentSpec], strict: bool
) -> PreflightReport:
    rep = PreflightReport()

    def problem(msg: str, hard: bool = False) -> None:
        if strict or hard:
            raise PreflightError(msg)
        rep.warnings.append(msg)

    wallet = await rest.wallet()
    if wallet is None:
        raise PreflightError("no UNIFIED wallet found")
    rep.equity = float(wallet.get("totalEquity") or 0.0)
    if rep.equity < cfg.challenge.starting_capital:
        problem(f"wallet equity {rep.equity:.2f} < starting capital {cfg.challenge.starting_capital:.2f}")

    symbols = set(cfg.challenge.symbols)
    open_pos = [
        f"{p['symbol']}:{p.get('side')}:{p.get('size')}"
        for p in await rest.positions()
        if p.get("symbol") in symbols and float(p.get("size") or 0.0) != 0.0
    ]
    if open_pos:
        problem(f"account is not flat on challenge symbols: {open_pos}", hard=True)

    info = await rest.account_info()
    rep.margin_mode = str(info.get("marginMode", ""))
    if rep.margin_mode and rep.margin_mode != "REGULAR_MARGIN":
        problem(f"margin mode {rep.margin_mode} != REGULAR_MARGIN (cross); liquidation model assumes cross")

    await rest.set_one_way_mode("USDT")
    target = max(1.0, math.ceil(cfg.risk.max_gross_leverage))
    for sym in sorted(symbols):
        lev = min(instruments[sym].max_leverage, target)
        await rest.set_leverage(sym, lev)
        rep.leverage[sym] = lev
    return rep
