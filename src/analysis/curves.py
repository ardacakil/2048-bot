"""Learning curves of the n-tuple training runs, read from the CSV logs (one row per 1,000 games).

    from analysis import curves
    figs, table = curves.make_all()      # all figures -> figures/*.png + *.svg, plus a summary table

Every figure draws all runs on shared axes with one fixed colour per run (run_styles). The
1,000-game rows are noisy, so the raw series is drawn faintly and a smoothed one on top:

  smooth      trailing rolling mean over `window` rows (20 rows = the last 20k games), so the
              value at game N only uses games <= N; the first window-1 rows are NaN
  log_binned  for log-x plots: mean of the rows in log-spaced bins of games, because a fixed
              20k-game window would blur everything below ~20k games

Milestones (games at which training saved a permanent checkpoint) are thin vertical lines on
every plot with games on the x-axis.
"""

import functools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

from ntuple.train import LOG_ROOT, REACH_TILES, ROOT

RUNS = ("overnight-a01", "overnight-a005", "overnight-a0025")
MILESTONES = (1_000, 10_000, 100_000, 1_000_000, 5_000_000)
FIGURE_ROOT = ROOT / "figures"
WINDOW = 20

# categorical slots in fixed order (validated colour-blind-safe); runs never get generated hues
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RUN_LABELS = {"overnight-a01": "α = 0.1", "overnight-a005": "α = 0.05", "overnight-a0025": "α = 0.025"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RAW_ALPHA = 0.18

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.constrained_layout.use": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.titlepad": 22,  # room for two rows of milestone labels above the axes
    "axes.titlecolor": INK_2,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "figure.titlesize": 12,
}


# ---------------------------------------------------------------- data


def run_styles(runs):
    """{run: {"color", "label"}}. The three overnight runs keep slots 0-2 whatever subset is
    plotted; other runs take the next free slots in order of appearance."""
    styles = {}
    free = [c for i, c in enumerate(SERIES_COLORS) if i >= len(RUNS)]
    for run in runs:
        if run in RUNS:
            color = SERIES_COLORS[RUNS.index(run)]
        elif free:
            color = free.pop(0)
        else:
            raise ValueError(f"more than {len(SERIES_COLORS)} runs: facet instead of adding colours")
        styles[run] = {"color": color, "label": RUN_LABELS.get(run, run)}
    return styles


def load_runs(runs=RUNS, log_root=LOG_ROOT):
    """{run: training log DataFrame}, with an extra `hours` column (elapsed_s / 3600)."""
    logs = {}
    for run in runs:
        df = pd.read_csv(Path(log_root) / f"{run}.csv")
        df["hours"] = df["elapsed_s"] / 3600
        logs[run] = df
    return logs


def smooth(s, window=WINDOW):
    """Trailing rolling mean over `window` rows; NaN until a full window is available."""
    return s.rolling(window, min_periods=window).mean()


def log_binned(df, col, bins_per_decade=20, x="games"):
    """Mean of `x` and `col` over rows grouped into log-spaced bins of `x` (empty bins dropped)."""
    # epsilon: log10(1000) = 2.9999999999999996 must land in the bin starting at 1000
    k = np.floor(np.log10(df[x].to_numpy(np.float64)) * bins_per_decade + 1e-9).astype(np.int64)
    return df.groupby(k)[[x, col]].mean().reset_index(drop=True)


def summary_table(logs, window=WINDOW, tile=16384, threshold=20.0):
    """One row per run: averages over the last `window` rows, plus the first game count (and
    hour) at which the trailing `window`-row mean of the `tile` reach rate exceeds `threshold`%."""
    cross_games = f"games_{tile}_gt_{threshold:g}pct"
    cross_hours = f"hours_{tile}_gt_{threshold:g}pct"
    rows = []
    for run, df in logs.items():
        tail = df.tail(window)
        crossed = df[smooth(df[f"reach_{tile}"], window) > threshold]
        rows.append({
            "run": run,
            "alpha": df["alpha"].iloc[-1],
            "games": int(df["games"].iloc[-1]),
            "hours": df["hours"].iloc[-1],
            "avg_score": tail["avg_score"].mean(),
            **{f"reach_{t}": tail[f"reach_{t}"].mean() for t in REACH_TILES},
            cross_games: int(crossed["games"].iloc[0]) if len(crossed) else None,
            cross_hours: crossed["hours"].iloc[0] if len(crossed) else None,
        })
    table = pd.DataFrame(rows).set_index("run")
    table.attrs.update(window=window, tile=tile, threshold=threshold, cross=(cross_games, cross_hours))
    return table


def format_summary(table):
    """Plain-text version of summary_table for printing."""
    a = table.attrs
    cross, cross_h = a["cross"]
    head = (f"{'run':16s} {'alpha':>6s} {'games':>11s} {'hours':>6s} {'avg score':>10s} "
            + " ".join(f"{t:>6d}" for t in REACH_TILES) + f"   {a['tile']} > {a['threshold']:g}% first at")
    lines = [f"last {a['window']} rows ({a['window']}k games); reach = % of games reaching the tile", head]
    for run, r in table.iterrows():
        first = "never" if pd.isna(r[cross]) else f"{int(r[cross]):,} games ({r[cross_h]:.2f} h)"
        lines.append(
            f"{run:16s} {r['alpha']:>6g} {int(r['games']):>11,} {r['hours']:>6.2f} {r['avg_score']:>10,.0f} "
            + " ".join(f"{r[f'reach_{t}']:>5.1f}%" for t in REACH_TILES) + f"   {first}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- figure helpers


def _styled(fn):
    """Build the figure under the shared rcParams (fonts, colours, grid).

    Only the RC keys are saved and restored. plt.rc_context would restore every rcParam, and the
    first figure of a notebook kernel resolves the inline backend, which turns interactive mode
    on; rolling that back stops every later figure from being displayed.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        old = {key: matplotlib.rcParams[key] for key in RC}
        matplotlib.rcParams.update(RC)
        try:
            return fn(*args, **kwargs)
        finally:
            matplotlib.rcParams.update(old)
    return wrapper


def format_count(x, _pos=None):
    """Tick label for counts: 0, 500, 10k, 2.5M."""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:g}{suffix}"
    return f"{x:g}"


def _games_axis(ax, logx=False):
    if logx:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_count))
    ax.set_xlabel("games trained")


def _score_axis(ax, label="score"):
    ax.yaxis.set_major_formatter(FuncFormatter(format_count))
    ax.set_ylabel(label)


def _to_axes(ax):
    """Data -> axes-fraction transform for the final limits."""
    ax.get_xlim(), ax.get_ylim()  # applies pending autoscaling, which transData does not
    return ax.transData + ax.transAxes.inverted()


def _milestones(ax, logs, milestones=MILESTONES, min_gap=0.04):
    """Thin vertical lines at the milestones the runs reached. Milestones closer than `min_gap`
    (axes fraction) share one label, e.g. "1k–100k" on a linear axis where they all sit at x ~ 0."""
    last_game = max(df["games"].iloc[-1] for df in logs.values())
    milestones = [m for m in milestones if m <= last_game]
    for m in milestones:
        ax.axvline(m, color=AXIS, lw=0.8, zorder=0.5, gid=f"milestone:{m}")
    to_axes = _to_axes(ax)
    groups = []
    for m in milestones:
        xa = to_axes.transform((m, 0))[0]
        if groups and xa - groups[-1][0] < min_gap:
            groups[-1][2] = m
        else:
            groups.append([xa, m, m])
    # a label that would overlap the previous one moves up to a second row
    renderer = ax.figure.canvas.get_renderer()
    row_right = [-np.inf, -np.inf]  # right edge (pixels) of the last label in each row
    for _, first, last in groups:
        label = format_count(first) if first == last else f"{format_count(first)}–{format_count(last)}"
        for row in (0, 1):
            t = ax.text(first, 1.0, f" {label}", color=MUTED, fontsize=7, ha="left", va="bottom",
                        transform=offset_copy(ax.get_xaxis_transform(), ax.figure, y=9 * row,
                                              units="points"), gid=f"milestone-label:{label}")
            bb = t.get_window_extent(renderer)
            if bb.x0 >= row_right[row] or row == 1:
                row_right[row] = bb.x1
                break
            t.remove()


def _end_labels(ax, ends, styles, min_gap=0.06):
    """Direct labels in the right margin, joined by a thin leader in the run's colour to the
    end of its line. `ends` = {run: (x, y)} in data coordinates. Call after limits are final."""
    to_axes = _to_axes(ax)
    runs = [r for r in ends if np.isfinite(ends[r][1])]
    ya = np.array([to_axes.transform(ends[r])[1] for r in runs])
    order = np.argsort(ya)
    pos = ya.copy()
    for i, j in zip(order[:-1], order[1:]):  # push overlapping labels up ...
        pos[j] = max(pos[j], pos[i] + min_gap)
    pos[order[-1]] = min(pos[order[-1]], 1.0)
    for j, i in zip(order[::-1][1:], order[::-1][:-1]):  # ... but not above the top of the axes
        pos[j] = min(pos[j], pos[i] - min_gap)
    for run, y in zip(runs, pos):
        ax.annotate(styles[run]["label"], xy=ends[run], xytext=(1.02, y), textcoords="axes fraction",
                    color=INK_2, fontsize=8, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=styles[run]["color"], lw=0.8,
                                    shrinkA=0, shrinkB=2))


def _legend(fig, styles):
    handles = [plt.Line2D([], [], color=s["color"], lw=2, label=s["label"]) for s in styles.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8,
               handlelength=1.6, columnspacing=1.4)


def _plot_series(ax, x, y, style, run, window, raw=True, agg="mean"):
    """Faint raw series + trailing rolling `agg` ("mean" = smooth()) on top. Returns the last
    finite (x, y) of the latter."""
    if raw:
        ax.plot(x, y, color=style["color"], lw=0.6, alpha=RAW_ALPHA, rasterized=True,
                gid=f"raw:{run}")
    ys = y.rolling(window, min_periods=window).agg(agg)
    ax.plot(x, ys, color=style["color"], gid=f"smooth:{run}")
    ok = np.isfinite(ys.to_numpy())
    return (x[ok].iloc[-1], ys[ok].iloc[-1]) if ok.any() else (np.nan, np.nan)


def _title(fig, title):
    fig.suptitle(title, x=0.01, ha="left", color=INK)


# ---------------------------------------------------------------- figures


@_styled
def plot_avg_score(logs, window=WINDOW, logx=False, bins_per_decade=20):
    """Average score per 1,000 games vs games trained (linear or log x)."""
    styles = run_styles(logs)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ends = {}
    for run, df in logs.items():
        if logx:
            ax.plot(df["games"], df["avg_score"], color=styles[run]["color"], lw=0.6,
                    alpha=RAW_ALPHA, rasterized=True, gid=f"raw:{run}")
            b = log_binned(df, "avg_score", bins_per_decade)
            ax.plot(b["games"], b["avg_score"], color=styles[run]["color"], gid=f"smooth:{run}")
            ends[run] = (b["games"].iloc[-1], b["avg_score"].iloc[-1])
        else:
            ends[run] = _plot_series(ax, df["games"], df["avg_score"], styles[run], run, window)
    _games_axis(ax, logx)
    _score_axis(ax, "average score")
    ax.set_ylim(bottom=0)
    smoothing = (f"log-binned means, {bins_per_decade} bins per decade" if logx
                 else f"trailing {window}k-game mean")
    ax.set_title(f"faint: each 1,000-game window · solid: {smoothing}")
    _title(fig, "Average score vs games trained" + (" (log scale)" if logx else ""))
    _milestones(ax, logs)
    _end_labels(ax, ends, styles)
    _legend(fig, styles)
    return fig


@_styled
def plot_reach_rates(logs, window=WINDOW, tiles=REACH_TILES):
    """One panel per tile: % of games reaching it, vs games trained."""
    styles = run_styles(logs)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, tile in zip(axes.flat, tiles):
        ends = {run: _plot_series(ax, df["games"], df[f"reach_{tile}"], styles[run], run, window)
                for run, df in logs.items()}
        _games_axis(ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of games")
        ax.set_title(f"reaching {tile}", color=INK, fontsize=10)
        _milestones(ax, logs)
        _end_labels(ax, ends, styles)
    for ax in axes[0]:
        ax.set_xlabel("")
    _title(fig, f"Tile-reach rates vs games trained (trailing {window}k-game mean)")
    _legend(fig, styles)
    return fig


@_styled
def plot_avg_moves(logs, window=WINDOW):
    """Average game length (moves) vs games trained."""
    styles = run_styles(logs)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ends = {run: _plot_series(ax, df["games"], df["avg_moves"], styles[run], run, window)
            for run, df in logs.items()}
    _games_axis(ax)
    _score_axis(ax, "moves per game")
    ax.set_ylim(bottom=0)
    ax.set_title(f"faint: each 1,000-game window · solid: trailing {window}k-game mean")
    _title(fig, "Average game length vs games trained")
    _milestones(ax, logs)
    _end_labels(ax, ends, styles)
    _legend(fig, styles)
    return fig


@_styled
def plot_max_score(logs, window=WINDOW):
    """Best score per 1,000 games, its rolling median over `window` rows, and the running max.

    The median of the per-window maxima shows the ceiling (~385k: the most a game can score
    without a second 16384 tile); a rolling max would follow the rare 400k+ outliers instead."""
    styles = run_styles(logs)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ends = {}
    for run, df in logs.items():
        ends[run] = _plot_series(ax, df["games"], df["max_score"], styles[run], run, window, agg="median")
        ax.plot(df["games"], df["max_score"].cummax(), color=styles[run]["color"], lw=0.8,
                alpha=0.6, gid=f"cummax:{run}")
    _games_axis(ax)
    _score_axis(ax, "max score")
    ax.set_ylim(bottom=0)
    ax.set_title(f"faint: max of each 1,000-game window · solid: trailing {window}-window median"
                 " of those maxima · thin: running max")
    _title(fig, "Max score vs games trained")
    _milestones(ax, logs)
    _end_labels(ax, ends, styles)
    _legend(fig, styles)
    return fig


@_styled
def plot_score_vs_hours(logs, window=WINDOW):
    """Smoothed average score vs wall-clock hours (the runs had different throughput)."""
    styles = run_styles(logs)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ends = {run: _plot_series(ax, df["hours"], df["avg_score"], styles[run], run, window, raw=False)
            for run, df in logs.items()}
    ax.set_xlabel("wall-clock hours")
    _score_axis(ax, "average score")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"trailing {window}k-game mean")
    _title(fig, "Average score vs training time")
    _end_labels(ax, ends, styles)
    _legend(fig, styles)
    return fig


FIGURES = {
    "curves_avg_score_vs_games": lambda logs, w: plot_avg_score(logs, w),
    "curves_avg_score_vs_games_logx": lambda logs, w: plot_avg_score(logs, w, logx=True),
    "curves_reach_rates_vs_games": plot_reach_rates,
    "curves_avg_moves_vs_games": plot_avg_moves,
    "curves_max_score_vs_games": plot_max_score,
    "curves_avg_score_vs_hours": plot_score_vs_hours,
}


def save_figure(fig, name, out_dir=FIGURE_ROOT, formats=("png", "svg"), dpi=200):
    """Write out_dir/name.<fmt> for each format; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.{fmt}" for fmt in formats]
    for p in paths:
        fig.savefig(p, dpi=dpi)
    return paths


def make_all(runs=RUNS, window=WINDOW, out_dir=FIGURE_ROOT, log_root=LOG_ROOT):
    """Draw and save every figure; returns ({name: Figure}, summary_table)."""
    logs = load_runs(runs, log_root)
    figs = {}
    for name, plot in FIGURES.items():
        figs[name] = plot(logs, window)
        save_figure(figs[name], name, out_dir)
    return figs, summary_table(logs, window)
