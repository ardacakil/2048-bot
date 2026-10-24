"""Evaluation report: summary table, run comparison and figures from results/eval/.

    uv run python -m analysis.report                      # rebuild results/eval_summary.csv
    uv run python -m analysis.report --compare overnight-a005,overnight-a0025 [--label greedy]

Figures use the same style and run colours as curves.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis import curves as C
from analysis import stats as S
from ntuple import evaluate as V

# ordinal ramp for max-tile bins (light -> dark; the lightest step still clears 2:1 on the surface)
TILE_BIN_COLORS = ("#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b")
TILE_BIN_INK = (C.INK, C.INK, "#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffffff")


def load_results(eval_root=V.EVAL_ROOT):
    """All per-game rows under eval_root (every <label>.csv; bench.csv excluded)."""
    frames = [V.read_results(p) for p in sorted(Path(eval_root).glob("**/*.csv")) if p.name != "bench.csv"]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=V.CSV_FIELDS)
    return pd.concat(frames, ignore_index=True)


def load_bench(eval_root=V.EVAL_ROOT):
    frames = [pd.read_csv(p, dtype={"run": str, "checkpoint": str}, keep_default_na=False)
              for p in sorted(Path(eval_root).glob("**/bench.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def summary(eval_root=V.EVAL_ROOT):
    return S.summarize_results(load_results(eval_root), load_bench(eval_root))


def write_summary(eval_root=V.EVAL_ROOT, path=None):
    """Rebuild <eval_root>/../eval_summary.csv (results/eval_summary.csv by default)."""
    path = Path(path) if path else Path(eval_root).parent / "eval_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary(eval_root).to_csv(path, index=False)
    return path


def compare(run_a, run_b, label="greedy", checkpoint="final", eval_root=V.EVAL_ROOT, tile=16384):
    """Paired comparison of two runs' results for one policy (see stats.compare_runs)."""
    read = lambda run: V.read_results(Path(eval_root) / run / checkpoint / f"{label}.csv")
    a, b = read(run_a), read(run_b)
    if not len(a) or not len(b):
        raise FileNotFoundError(f"no {label} results for {run_a if not len(a) else run_b} ({checkpoint})")
    return S.compare_runs(a, b, run_a, run_b, tile)


# ---------------------------------------------------------------- figures


@C._styled
def plot_progression(table, policy="greedy", depth=0, tile=16384, runs=None):
    """Mean score and `tile` reach rate (95% CIs) against games trained, one point per milestone
    plus the final checkpoint (hollow marker) at its games_trained."""
    t = table[(table["policy"] == policy) & (table["depth"] == depth) & (table["run"] != "-")]
    runs = runs or [r for r in C.RUNS if r in set(t["run"])] + sorted(set(t["run"]) - set(C.RUNS))
    styles = C.run_styles(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (col, lo, hi, ylabel) in zip(axes, [
            ("mean_score", "mean_lo", "mean_hi", "mean score"),
            (f"reach_{tile}", f"reach_{tile}_lo", f"reach_{tile}_hi", f"% of games reaching {tile}")]):
        ends = {}
        for run in runs:
            g = t[t["run"] == run].sort_values("games_trained")
            if not len(g):
                continue
            c = styles[run]["color"]
            x, y = g["games_trained"].to_numpy(), g[col].to_numpy()
            err = np.vstack([y - g[lo].to_numpy(), g[hi].to_numpy() - y])
            ax.errorbar(x, y, yerr=err, color=c, lw=1.6, capsize=3, elinewidth=1, gid=f"line:{run}")
            final = (g["checkpoint"] == "final").to_numpy()
            ax.plot(x[~final], y[~final], "o", color=c, ms=6, mec=C.SURFACE, mew=1.5)
            ax.plot(x[final], y[final], "o", color=C.SURFACE, ms=6, mec=c, mew=1.8, gid=f"final:{run}")
            ends[run] = (x[-1], y[-1])
        C._games_axis(ax, logx=True)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        C._end_labels(ax, ends, styles)
    axes[0].yaxis.set_major_formatter(FuncFormatter(C.format_count))
    n = int(t["games"].min()) if len(t) else 0
    axes[0].set_title(f"{policy_name(policy, depth)}, {n:,} games per point · 95% CIs · hollow: final checkpoint")
    C._title(fig, "Strength vs games trained")
    C._legend(fig, styles)
    return fig


def policy_name(policy, depth):
    return V.policy_label(policy, depth)


@C._styled
def plot_max_tile_distribution(table, run, checkpoint="final", include_random=True):
    """100% stacked bars: share of games per max tile, one bar per policy."""
    t = table[(table["run"] == run) & (table["checkpoint"] == checkpoint)]
    if include_random:
        t = pd.concat([table[table["policy"] == "random"], t])
    t = t.sort_values(["policy", "depth"], key=lambda s: s.map(_policy_order) if s.name == "policy" else s)
    labels = [policy_name(p, d) for p, d in zip(t["policy"], t["depth"])]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(t) + 1.6))
    ax.grid(False)
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for b, color, ink in zip(S.MAX_TILE_BINS, TILE_BIN_COLORS, TILE_BIN_INK):
        w = t[f"max_tile_{S.bin_name(b)}"].to_numpy(np.float64)
        ax.barh(y, w, left=left, color=color, height=0.62, edgecolor=C.SURFACE, linewidth=1.5,
                label="≤ 1024" if b == 1024 else f"{b}", gid=f"bin:{b}")
        for yi, li, wi in zip(y, left, w):
            if wi >= 7:  # label only segments wide enough to hold it
                ax.text(li + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", color=ink, fontsize=8)
        left += w
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=C.INK_2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of games")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(side == "bottom")
    n = int(t["games"].min()) if len(t) else 0
    ax.set_title(f"{run} · {checkpoint} checkpoint · at least {n:,} games per bar")
    C._title(fig, "Largest tile reached")
    fig.legend(loc="outside lower center", ncols=len(S.MAX_TILE_BINS), fontsize=8, title="max tile",
               title_fontsize=8)
    return fig


def _policy_order(policy):
    return {"random": 0, "greedy": 1, "expectimax": 2}[policy]


@C._styled
def plot_strength_vs_cost(table, run, checkpoint="final"):
    """Mean score (95% CI) against ms/move per policy; ms/move from --bench when available."""
    t = table[(table["run"] == run) & (table["checkpoint"] == checkpoint)]
    t = t.sort_values(["policy", "depth"], key=lambda s: s.map(_policy_order) if s.name == "policy" else s)
    bench = "ms_per_move_bench" in t and t["ms_per_move_bench"].notna().all()
    x = t["ms_per_move_bench" if bench else "ms_per_move"].to_numpy(np.float64)
    y = t["mean_score"].to_numpy(np.float64)
    err = np.vstack([y - t["mean_lo"].to_numpy(), t["mean_hi"].to_numpy() - y])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    color = C.SERIES_COLORS[0]
    ax.plot(x, y, color=color, lw=1.2, alpha=0.5)
    ax.errorbar(x, y, yerr=err, fmt="o", color=color, ms=7, mec=C.SURFACE, mew=1.5, capsize=3,
                elinewidth=1)
    for xi, yi, p, d in zip(x, y, t["policy"], t["depth"]):
        ax.annotate(policy_name(p, d), (xi, yi), xytext=(6, -12), textcoords="offset points",
                    color=C.INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("ms per move" + (" (single process)" if bench else " (per worker, parallel)"))
    C._score_axis(ax, "mean score")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{run} · {checkpoint} checkpoint · 95% CIs")
    C._title(fig, "Strength vs search cost")
    return fig


# ---------------------------------------------------------------- replays in the notebook


def replay_viewer(path):
    """ipywidgets slider stepping through a replay file (board, score, next move)."""
    import ipywidgets as widgets
    from IPython.display import HTML, display

    from game import engine as E
    from game import replay as R

    doc = R.load(path)
    steps = doc["steps"]
    boards = R.trajectory(doc["init"], [s[0] for s in steps], [s[2:] for s in steps])
    scores = np.concatenate([[0], np.cumsum([s[1] for s in steps])])
    m = doc["meta"]
    print(f"{m['run']} {m['checkpoint']} {V.policy_label(m['policy'], m['depth'])} seed {m['seed']}: "
          f"score {m['score']:,}, max tile {m['max_tile']:,}, {m['moves']:,} moves ({m['selection']})")

    def show(step=0):
        nxt = E.ACTION_NAMES[steps[step][0]] if step < len(steps) else "game over"
        display(HTML(f"<pre>move {step:,}   score {scores[step]:,}   next: {nxt}\n"
                     f"{E.format_board(int(boards[step]))}</pre>"))

    return widgets.interact(show, step=widgets.IntSlider(0, 0, len(steps), 1, layout={"width": "600px"}))


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluation summary and run comparison.")
    p.add_argument("--eval-root", type=Path, default=V.EVAL_ROOT)
    p.add_argument("--compare", help="RUN_A,RUN_B: paired comparison of their results")
    p.add_argument("--label", default="greedy", help="policy label for --compare (e.g. expectimax-d2)")
    p.add_argument("--checkpoint", default="final")
    args = p.parse_args(argv)
    if args.compare:
        a, b = args.compare.split(",")
        print(S.format_comparison(compare(a, b, args.label, args.checkpoint, args.eval_root)))
    else:
        print(f"summary: {write_summary(args.eval_root)}")


if __name__ == "__main__":
    main()
