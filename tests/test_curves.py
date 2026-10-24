import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Annotation

from analysis import curves as C
from ntuple.train import LOG_FIELDS


def fake_log(n_rows, alpha, cross_row=None):
    """Synthetic training log: linear trends; reach_16384 jumps from 0 to 50 at cross_row."""
    games = np.arange(1, n_rows + 1) * 1000
    i = np.arange(n_rows, dtype=np.float64)
    df = pd.DataFrame({
        "games": games,
        "moves": games * 500,
        "alpha": alpha,
        "avg_score": 1000 + 10 * i,
        "max_score": 5000 + 20 * i,
        "avg_moves": 400 + i,
        "reach_2048": np.minimum(i, 100.0),
        "reach_4096": np.minimum(i / 2, 100.0),
        "reach_8192": np.minimum(i / 4, 100.0),
        "reach_16384": np.where(i >= cross_row, 50.0, 0.0) if cross_row is not None else 0.0,
        "games_per_s": 1000.0,
        "moves_per_s": 5e5,
        "elapsed_s": games / 1000 * 3.6,  # 1,000 rows = 1 hour
    })
    return df[LOG_FIELDS]


@pytest.fixture
def log_root(tmp_path):
    for run, alpha, cross in zip(C.RUNS, (0.1, 0.05, 0.025), (30, 45, None)):
        fake_log(60, alpha, cross).to_csv(tmp_path / f"{run}.csv", index=False)
    return tmp_path


@pytest.fixture
def logs(log_root):
    return C.load_runs(C.RUNS, log_root)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


REACHED = [1000, 10000]  # milestones within the 60k games of the fake logs


def gids(ax, prefix):
    return sorted(l.get_gid() for l in ax.get_lines() if (l.get_gid() or "").startswith(prefix))


# ---------------------------------------------------------------- data


def test_load_runs_adds_hours(logs):
    assert list(logs) == list(C.RUNS)
    df = logs["overnight-a01"]
    assert df["hours"].iloc[-1] == pytest.approx(60 * 3.6 / 3600)
    assert list(df.columns) == LOG_FIELDS + ["hours"]


def test_smooth_is_trailing_mean_with_full_window():
    s = C.smooth(pd.Series(np.arange(30, dtype=np.float64)), window=5)
    assert s.iloc[:4].isna().all()
    assert s.iloc[4] == 2.0  # mean of 0..4
    np.testing.assert_allclose(s.iloc[4:], np.arange(4, 30) - 2.0)


def test_log_binned_bins_are_log_spaced_and_respect_powers_of_ten():
    df = pd.DataFrame({"games": [1000, 9000, 10000, 99000, 100000], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    b = C.log_binned(df, "y", bins_per_decade=1)
    # decades [1k, 10k), [10k, 100k), [100k, 1M): exact powers of ten open a new bin
    assert b["games"].tolist() == [5000, 54500, 100000]
    assert b["y"].tolist() == [1.5, 3.5, 5.0]


def test_log_binned_drops_empty_bins_and_is_monotone():
    df = pd.DataFrame({"games": np.arange(1, 201) * 1000, "y": np.arange(200, dtype=np.float64)})
    b = C.log_binned(df, "y", bins_per_decade=20)
    assert (np.diff(b["games"]) > 0).all() and (np.diff(b["y"]) > 0).all()
    assert len(b) <= 2 * 20 + 2  # 1k..200k spans < 2.4 decades, the first bins hold one row each
    assert b["games"].iloc[0] == 1000


def test_run_styles_fixed_per_run_and_never_cycled():
    full = C.run_styles(C.RUNS)
    subset = C.run_styles(["overnight-a0025", "overnight-a01"])
    for run in subset:  # colour follows the run, not its position in the list
        assert subset[run]["color"] == full[run]["color"]
    assert len({s["color"] for s in full.values()}) == 3
    assert full["overnight-a005"]["label"] == "α = 0.05"
    other = C.run_styles(["x", "overnight-a01", "y"])
    assert other["x"]["color"] == C.SERIES_COLORS[3] and other["y"]["color"] == C.SERIES_COLORS[4]
    assert other["x"]["label"] == "x"
    with pytest.raises(ValueError):
        C.run_styles([f"run{i}" for i in range(6)])  # only 5 slots left after the overnight runs


def test_summary_table(logs):
    t = C.summary_table(logs, window=20)
    a01 = t.loc["overnight-a01"]
    assert a01["games"] == 60_000 and a01["alpha"] == 0.1
    assert a01["avg_score"] == pytest.approx(1000 + 10 * np.arange(40, 60).mean())
    assert a01["reach_2048"] == pytest.approx(np.arange(40, 60).mean())
    # 16384 jumps to 50% at row 30: the trailing 20-row mean exceeds 20% once > 8 of the last
    # 20 rows are past the jump, i.e. first at row 38 = 39,000 games
    assert a01["games_16384_gt_20pct"] == 39_000
    assert a01["hours_16384_gt_20pct"] == pytest.approx(39 * 3.6 / 3600)
    assert t.loc["overnight-a005", "games_16384_gt_20pct"] == 54_000
    assert pd.isna(t.loc["overnight-a0025", "games_16384_gt_20pct"])


def test_summary_table_threshold_is_strict():
    df = fake_log(30, 0.1)
    df["reach_16384"] = 20.0  # equal to the threshold, never above it
    df["hours"] = df["elapsed_s"] / 3600
    t = C.summary_table({"r": df}, window=5)
    assert pd.isna(t.loc["r", "games_16384_gt_20pct"])


def test_format_summary(logs):
    text = C.format_summary(C.summary_table(logs))
    assert "39,000 games" in text and "never" in text
    assert text.count("\n") == 1 + len(C.RUNS)


@pytest.mark.parametrize("x, label", [(0, "0"), (500, "500"), (1000, "1k"), (2_500_000, "2.5M"),
                                      (1e7, "10M"), (150_000, "150k")])
def test_format_count(x, label):
    assert C.format_count(x) == label


# ---------------------------------------------------------------- figures


def test_avg_score_linear_has_raw_smoothed_and_milestones(logs):
    fig = C.plot_avg_score(logs)
    ax = fig.axes[0]
    assert gids(ax, "raw:") == sorted(f"raw:{r}" for r in C.RUNS)
    assert gids(ax, "smooth:") == sorted(f"smooth:{r}" for r in C.RUNS)
    assert gids(ax, "milestone:") == sorted(f"milestone:{m}" for m in REACHED)
    assert ax.get_xscale() == "linear"
    smooth = next(l for l in ax.get_lines() if l.get_gid() == "smooth:overnight-a01")
    assert np.isnan(smooth.get_ydata()[:19]).all()  # trailing window of 20 rows
    assert sum(isinstance(t, Annotation) for t in ax.texts) == 3  # one direct label per run


def test_avg_score_logx_uses_log_bins_and_milestones(logs):
    fig = C.plot_avg_score(logs, logx=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(gids(ax, "milestone:")) == len(REACHED)
    smooth = next(l for l in ax.get_lines() if l.get_gid() == "smooth:overnight-a01")
    assert len(smooth.get_xdata()) < 60  # binned, not one point per row


def test_reach_rates_one_panel_per_tile(logs):
    fig = C.plot_reach_rates(logs)
    assert len(fig.axes) == len(C.REACH_TILES)
    for ax, tile in zip(fig.axes, C.REACH_TILES):
        assert str(tile) in ax.get_title(loc="left")
        assert len(gids(ax, "smooth:")) == 3 and len(gids(ax, "milestone:")) == len(REACHED)
        assert ax.get_ylim() == (0, 100)


def test_max_score_draws_rolling_median_and_running_max(logs):
    ax = C.plot_max_score(logs).axes[0]
    rolling = next(l for l in ax.get_lines() if l.get_gid() == "smooth:overnight-a01")
    running = next(l for l in ax.get_lines() if l.get_gid() == "cummax:overnight-a01")
    y = logs["overnight-a01"]["max_score"].to_numpy()
    np.testing.assert_allclose(rolling.get_ydata()[19:], [np.median(y[i - 19:i + 1]) for i in range(19, 60)])
    np.testing.assert_allclose(running.get_ydata(), np.maximum.accumulate(y))


def test_avg_moves_and_hours_figures(logs):
    ax = C.plot_avg_moves(logs).axes[0]
    assert len(gids(ax, "smooth:")) == 3 and len(gids(ax, "milestone:")) == len(REACHED)
    ax = C.plot_score_vs_hours(logs).axes[0]
    assert gids(ax, "raw:") == [] and gids(ax, "milestone:") == []
    line = next(l for l in ax.get_lines() if l.get_gid() == "smooth:overnight-a01")
    np.testing.assert_allclose(line.get_xdata(), logs["overnight-a01"]["hours"])


def test_end_labels_do_not_overlap():
    # a005 and a0025 end almost on top of each other in the real logs
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 100])
    ends = {"overnight-a01": (1, 50.0), "overnight-a005": (1, 80.0), "overnight-a0025": (1, 80.5)}
    C._end_labels(ax, ends, C.run_styles(C.RUNS), min_gap=0.06)
    ys = sorted(t.xyann[1] for t in ax.texts)
    assert np.diff(ys).min() >= 0.06 - 1e-12


def test_milestone_labels_merge_close_lines_and_never_overlap(logs):
    long = {r: fake_log(14_000, 0.1) for r in C.RUNS}  # 14M games: milestones like the real runs
    ax = C.plot_avg_score(long).axes[0]
    labels = [t for t in ax.texts if (t.get_gid() or "").startswith("milestone-label:")]
    assert [t.get_gid().split(":")[1] for t in labels] == ["1k–100k", "1M", "5M"]
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in labels]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b)


def test_save_figure_writes_png_and_svg(tmp_path, logs):
    paths = C.save_figure(C.plot_avg_moves(logs), "moves", tmp_path / "figs")
    assert [p.name for p in paths] == ["moves.png", "moves.svg"]
    assert paths[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths[1].read_text().lstrip().startswith("<?xml")


def test_make_all_writes_every_figure(tmp_path, log_root):
    figs, table = C.make_all(out_dir=tmp_path / "figs", log_root=log_root)
    assert list(figs) == list(C.FIGURES)
    assert len(C.FIGURES) == 6
    written = sorted(p.name for p in (tmp_path / "figs").iterdir())
    assert written == sorted(f"{n}.{ext}" for n in C.FIGURES for ext in ("png", "svg"))
    assert list(table.index) == list(C.RUNS)
