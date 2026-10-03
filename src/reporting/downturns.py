"""How each strategy behaved in the major downturns.

For each configured range, SPY's deepest peak-to-recovery episode whose trough
falls in the range defines the downturn. Each strategy is measured from SPY's
peak to SPY's recovery (or the end of the data):

* drawdown: the strategy's largest peak-to-trough fall within that period;
* recovery: days from that drawdown's peak until the strategy regained it;
* exits / re-entries: days when risk exposure (everything except cash and,
  for GEM, bonds) fell / rose by at least ``exposure_change``;
* whipsaw: an exit followed by a re-entry while SPY's price was higher than
  at the exit, i.e. it sold and then bought back higher;
* label: "Avoided" if the drawdown was at most ``avoided_max_ratio`` of SPY's,
  plus "whipsawed" if any whipsaw occurred; otherwise "Whipsawed" or "Neither".
"""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import CASH
from src.backtest.metrics import drawdown_episodes
from src.config import heldout_start

DEFENSIVE_PARAMS = ("cash", "bonds")

DOWNTURN_NAMES = {
    "dotcom_2000_2002": "2000–2002 dot-com bear market",
    "gfc_2008_2009": "2008–2009 financial crisis",
    "covid_2020": "2020 COVID crash and recovery",
    "bear_2022": "2022 bear market",
}


def downturn_name(key: str) -> str:
    return DOWNTURN_NAMES.get(key, key.replace("_", " "))


def spy_episode(spy_equity: pd.Series, search_start, search_end) -> pd.Series | None:
    """SPY's deepest drawdown episode whose trough lies in [search_start, search_end]."""
    ep = drawdown_episodes(spy_equity)
    inside = ep[(ep["trough"] >= pd.Timestamp(search_start)) & (ep["trough"] <= pd.Timestamp(search_end))]
    if inside.empty:
        return None
    return inside.loc[inside["depth"].idxmin()]


def defensive_symbols(params: dict) -> set[str]:
    return {params[k] for k in DEFENSIVE_PARAMS if params.get(k)} | {CASH}


def exposure(weights: pd.DataFrame, defensive: set[str]) -> pd.Series:
    risk_cols = [c for c in weights.columns if c not in defensive]
    return weights[risk_cols].sum(axis=1)


def exits_and_whipsaws(expo: pd.Series, spy_price: pd.Series, threshold: float) -> tuple[int, int, int]:
    """Count exits, re-entries and whipsaws (exit then re-entry at a higher SPY price)."""
    d = expo.diff().fillna(0.0)
    exits = list(d.index[d <= -threshold])
    entries = list(d.index[d >= threshold])
    whipsaws = 0
    for x in exits:
        later = [e for e in entries if e > x]
        if later and spy_price.loc[later[0]] > spy_price.loc[x]:
            whipsaws += 1
    return len(exits), len(entries), whipsaws


def _top_holdings(w: pd.Series, n: int = 4) -> str:
    w = w[w > 0.005].sort_values(ascending=False).head(n)
    return ", ".join(f"{k} {v:.0%}" for k, v in w.items()) if len(w) else "none"


def _recovery(equity: pd.Series, peak_date: pd.Timestamp, trough_date: pd.Timestamp):
    peak_val = equity.loc[peak_date]
    after = equity[(equity.index > trough_date) & (equity >= peak_val)]
    return (after.index[0], (after.index[0] - peak_date).days) if len(after) else (None, None)


def downturn_table(runs: dict, market, settings: dict, strategy_params: dict[str, dict]) -> pd.DataFrame:
    cfg = settings["report"]
    labels = cfg["downturn_labels"]
    spy_equity = runs["benchmark"].pretax.equity
    spy_price = market.prices[settings["benchmark"]]
    locked = not settings["dates"].get("heldout_unlocked", False)
    rows = []
    for dname, (s0, s1) in cfg["downturns"].items():
        if locked and pd.Timestamp(s1) >= heldout_start(settings):
            rows.append({"downturn": dname, "strategy": None, "status": "held out (locked)"})
            continue
        ep = spy_episode(spy_equity, s0, s1)
        if ep is None:
            rows.append({"downturn": dname, "strategy": None, "status": "no SPY drawdown found in data"})
            continue
        peak, trough = ep["peak"], ep["trough"]
        end = ep["recovery"] if pd.notna(ep["recovery"]) else spy_equity.index[-1]
        base = {"downturn": dname, "spy_peak": peak.date(), "spy_trough": trough.date(),
                "spy_recovery": ep["recovery"].date() if pd.notna(ep["recovery"]) else None,
                "spy_drawdown": float(ep["depth"]), "spy_days_to_recover": int(ep["days_to_recover"]),
                "spy_recovered": bool(ep["recovered"])}
        for name, run in runs.items():
            if name == "benchmark":
                continue
            eq = run.pretax.equity
            if eq.index[0] > trough:
                rows.append({**base, "strategy": name, "status": "not live"})
                continue
            window = eq[(eq.index >= max(peak, eq.index[0])) & (eq.index <= end)]
            ep_s = drawdown_episodes(window)
            if ep_s.empty:
                dd, s_peak, s_trough = 0.0, window.index[0], window.index[0]
                rec_date, rec_days = window.index[0], 0
            else:
                worst = ep_s.loc[ep_s["depth"].idxmin()]
                dd, s_peak, s_trough = float(worst["depth"]), worst["peak"], worst["trough"]
                rec_date, rec_days = _recovery(eq, s_peak, s_trough)
            wts = run.pretax.weights
            expo = exposure(wts.loc[window.index], defensive_symbols(strategy_params.get(name, {})))
            n_exit, n_entry, n_whip = exits_and_whipsaws(expo, spy_price.reindex(window.index),
                                                         labels["exposure_change"])
            avoided = abs(dd) <= labels["avoided_max_ratio"] * abs(base["spy_drawdown"])
            label = ("Avoided, but whipsawed" if avoided and n_whip else "Avoided" if avoided
                     else "Whipsawed" if n_whip else "Neither")
            into_date = window.index[0]
            during = wts.loc[(wts.index >= into_date) & (wts.index <= trough)].mean()
            rows.append({
                **base, "strategy": name,
                "status": "ok" if eq.index[0] <= peak + pd.Timedelta(days=7) else "partial",
                "live_from": into_date.date(),
                "drawdown": dd, "drawdown_peak": s_peak.date(), "drawdown_trough": s_trough.date(),
                "recovered_on": rec_date.date() if rec_date is not None else None,
                "days_to_recover": rec_days,
                "exits": n_exit, "reentries": n_entry, "whipsaws": n_whip, "label": label,
                "holdings_into": _top_holdings(wts.loc[into_date].drop(CASH, errors="ignore")),
                "holdings_during": _top_holdings(during.drop(CASH, errors="ignore")),
            })
    return pd.DataFrame(rows)


def describe_downturn_row(row: pd.Series, display) -> str:
    """One plain-English sentence for a strategy in a downturn."""
    name = display(row["strategy"])
    if row["status"] == "not live":
        return f"{name} had not started trading yet."
    rec = (f"recovered its prior peak after {int(row['days_to_recover'])} days" if pd.notna(row.get("days_to_recover"))
           else "had not recovered by the end of the in-sample data")
    spy_rec = (f"{int(row['spy_days_to_recover'])} days" if row["spy_recovered"] else "not recovered by the end of the data")
    verdict = {
        "Avoided": "It avoided most of this downturn.",
        "Avoided, but whipsawed": "It avoided most of this downturn, though at least once it sold and bought back higher.",
        "Whipsawed": "It did not avoid the decline and was whipsawed: it sold and then bought back higher.",
        "Neither": "It neither avoided the decline nor was whipsawed.",
    }[row["label"]]
    return (f"{name} fell {abs(row['drawdown']):.0%} (SPY: {abs(row['spy_drawdown']):.0%}, recovery {spy_rec}) and "
            f"{rec}. {verdict} Exits to defensive assets: {int(row['exits'])}; re-entries: {int(row['reentries'])}.")
