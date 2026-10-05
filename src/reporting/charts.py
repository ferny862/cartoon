"""Report charts (matplotlib, rendered to base64 PNG in a light and a dark theme).

Design rules:
* small multiples: one panel per strategy, each against SPY over the same period
* each strategy keeps one categorical color on every chart; SPY is neutral ink
* thin lines, hairline grid, no top/right spines, one y-axis per panel
* allocation uses a single-hue sequential ramp (weights), not categorical hues

The categorical slots are the validated reference palette (validator: all
checks pass in both modes; light-mode slots 3-5 are below 3:1 contrast, so
each panel is titled and every chart has a table twin in the report).
"""

from __future__ import annotations

import base64
import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from src.backtest.engine import CASH  # noqa: E402
from src.backtest.metrics import calendar_year_returns, drawdown_series, rolling_excess_return  # noqa: E402
from src.reporting.names import SLOT, display_name  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "slots": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "seq": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "slots": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "seq": ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
TIMES = FuncFormatter(lambda v, _: f"{v:g}x")


def color_for(name: str, theme: dict) -> str:
    base = name.replace("__french", "")
    return theme["slots"][SLOT[base]] if base in SLOT else theme["secondary"]


def _style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    ax.title.set_color(theme["ink"])


def _grid(n: int, theme: dict, ncols: int = 2, panel_h: float = 2.7, width: float = 11.0):
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, panel_h * nrows), squeeze=False)
    fig.patch.set_facecolor(theme["surface"])
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax in flat[:n]:
        _style(ax, theme)
    return fig, flat[:n]


def _legend(ax, theme: dict, loc: str = "upper left") -> None:
    leg = ax.legend(loc=loc, fontsize=8, frameon=False)
    for text in leg.get_texts():
        text.set_color(theme["secondary"])


def _log_axis(ax) -> None:
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    # Narrow ranges (e.g. a 5-year held-out window) need finer ticks to be readable.
    subs = (1.0, 2.0, 5.0) if hi / max(lo, 1e-9) > 4 else (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(TIMES)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    if not getattr(fig, "_manual_layout", False):
        fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def both_themes(draw, *args, **kwargs) -> dict[str, str]:
    """Render ``draw(theme, ...)`` in the light and dark themes."""
    return {mode: _to_b64(draw(THEMES[mode], *args, **kwargs)) for mode in ("light", "dark")}


def _panel_title(ax, name: str, theme: dict, extra: str = "") -> None:
    ax.set_title(display_name(name) + extra, fontsize=9.5, loc="left", color=theme["ink"])


# --------------------------------------------------------------------- charts

def equity_chart(theme: dict, runs: dict, benchmark_label: str = "SPY, same period"):
    names = [n for n in runs if n != "benchmark"]
    fig, axes = _grid(len(names), theme)
    for ax, name in zip(axes, names):
        run = runs[name]
        cap = run.pretax.initial_capital
        ax.plot(run.benchmark.equity / cap, color=theme["secondary"], lw=1.1, label=benchmark_label)
        ax.plot(run.pretax.equity / cap, color=color_for(name, theme), lw=1.6, label=display_name(name))
        _log_axis(ax)
        _panel_title(ax, name, theme, "  ·  growth of $1 (log)")
        _legend(ax, theme)
    return fig


def drawdown_chart(theme: dict, runs: dict):
    names = [n for n in runs if n != "benchmark"]
    fig, axes = _grid(len(names), theme)
    for ax, name in zip(axes, names):
        run = runs[name]
        ax.plot(drawdown_series(run.benchmark.equity), color=theme["secondary"], lw=1.0, label="SPY, same period")
        ax.plot(drawdown_series(run.pretax.equity), color=color_for(name, theme), lw=1.4, label=display_name(name))
        ax.yaxis.set_major_formatter(PCT)
        ax.axhline(0, color=theme["axis"], lw=0.8)
        _panel_title(ax, name, theme, "  ·  drawdown")
        _legend(ax, theme, "lower left")
    return fig


def rolling_excess_chart(theme: dict, runs: dict, years: int = 3):
    names = [n for n in runs if n != "benchmark"]
    fig, axes = _grid(len(names), theme)
    for ax, name in zip(axes, names):
        run = runs[name]
        ex = rolling_excess_return(run.pretax.equity, run.benchmark.equity, years)
        ax.axhline(0, color=theme["axis"], lw=0.8)
        if len(ex):
            ax.plot(ex, color=color_for(name, theme), lw=1.5)
            ax.yaxis.set_major_formatter(PCT)
        else:
            ax.text(0.5, 0.5, f"Less than {years} years of history", ha="center", va="center",
                    transform=ax.transAxes, color=theme["muted"], fontsize=9)
        _panel_title(ax, name, theme, f"  ·  {years}-yr excess vs SPY, per yr")
    return fig


def annual_returns_chart(theme: dict, runs: dict):
    names = [n for n in runs if n != "benchmark"]
    fig, axes = _grid(len(names), theme)
    for ax, name in zip(axes, names):
        run = runs[name]
        cap = run.pretax.initial_capital
        s = calendar_year_returns(run.pretax.equity, base=cap)
        b = calendar_year_returns(run.benchmark.equity, base=cap)
        full = s.index[~s["partial"]]
        x = np.arange(len(full))
        ax.bar(x - 0.21, s.loc[full, "return"], width=0.4, color=color_for(name, theme), label=display_name(name))
        ax.bar(x + 0.21, b.loc[full, "return"], width=0.4, color=theme["secondary"], label="SPY")
        ax.axhline(0, color=theme["axis"], lw=0.8)
        step = max(1, len(full) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(y) for y in full[::step]])
        ax.yaxis.set_major_formatter(PCT)
        _panel_title(ax, name, theme, "  ·  calendar years")
        _legend(ax, theme)
    return fig


def allocation_chart(theme: dict, runs: dict):
    names = [n for n in runs if n != "benchmark"]
    fig, axes = _grid(len(names), theme, panel_h=3.0)
    cmap = LinearSegmentedColormap.from_list("weights", theme["seq"])
    image = None
    for ax, name in zip(axes, names):
        w = runs[name].pretax.weights.resample("ME").mean()
        keep = [c for c in w.columns if w[c].max() > 0.01]
        w = w[keep]
        if CASH in w.columns:
            w = w.rename(columns={CASH: "uninvested"})
        image = ax.imshow(w.T.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
                          extent=(0, len(w), len(keep) - 0.5, -0.5))
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels(list(w.columns), fontsize=7)
        months = w.index
        ticks = [i for i in range(len(w)) if months[i].month == 1]      # January of each year
        step = max(1, math.ceil(len(ticks) / 7))
        ax.set_xticks([t + 0.5 for t in ticks[::step]])
        ax.set_xticklabels([str(months[i].year) for i in ticks[::step]])
        ax.grid(False)
        _panel_title(ax, name, theme, "  ·  monthly weights")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig._manual_layout = True
    if image is not None:
        cax = fig.add_axes((0.945, 0.2, 0.012, 0.6))
        bar = fig.colorbar(image, cax=cax, format=PCT)
        bar.ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
        bar.outline.set_visible(False)
    return fig


def french_chart(theme: dict, runs: dict):
    names = list(runs)
    fig, axes = _grid(len(names), theme)
    for ax, name in zip(axes, names):
        run = runs[name]
        cap = run.pretax.initial_capital
        ax.plot(run.benchmark.equity / cap, color=theme["secondary"], lw=1.1, label="French market, same period")
        ax.plot(run.pretax.equity / cap, color=color_for(name, theme), lw=1.5, label="Strategy (gross)")
        _log_axis(ax)
        _panel_title(ax, name, theme, "  ·  gross, non-investable")
        _legend(ax, theme)
    return fig
