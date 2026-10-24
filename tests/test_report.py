import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import report as RE
from analysis import stats as S
from ntuple import evaluate as V

from conftest import EVAL_RUN


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def fake_rows(run, checkpoint, games_trained, policy, depth, n, seed=0, shift=0):
    rng = np.random.default_rng(seed)
    tiles = rng.choice([1024, 2048, 4096, 8192, 16384, 32768, 65536], n, p=[.05, .1, .15, .3, .3, .08, .02])
    return V.typed(pd.DataFrame({
        "run": run, "checkpoint": checkpoint, "games_trained": games_trained, "policy": policy,
        "depth": depth, "seed": np.arange(n), "score": (tiles * 12 + rng.integers(0, 5000, n) + shift),
        "max_tile": tiles, "max_tile_pos": 0, "max_tile_in_corner": True,
        "corner_frac": rng.uniform(0.5, 0.9, n), "corner_stable_frac": rng.uniform(0.3, 0.5, n), "moves": 5000,
        "wall_s": 1.0, "ms_per_move": 0.2, "final_board": "0x" + "0" * 16, "n_workers": 8,
        "prune": 1e-4 if policy == "expectimax" else 0.0, "tt_bits": 20 if policy == "expectimax" else 0}))


@pytest.fixture
def table():
    parts = [fake_rows("-", "-", 0, "random", 0, 50)]
    for run, shift in (("overnight-a005", 0), ("overnight-a0025", 2000)):
        for ckpt, games in (("games-0000001000", 1000), ("games-0001000000", 1_000_000), ("final", 12_900_000)):
            parts.append(fake_rows(run, ckpt, games, "greedy", 0, 200, seed=games % 7, shift=shift))
    for d in (1, 2):
        parts.append(fake_rows("overnight-a005", "final", 12_900_000, "expectimax", d, 100, seed=d))
    return S.summarize_results(pd.concat(parts))


def test_load_and_write_summary(eval_root):
    df = RE.load_results(eval_root)
    assert len(df) == 20 + 20 + 8 and df["max_tile_in_corner"].dtype == bool
    path = RE.write_summary(eval_root)
    assert path == eval_root.parent / "eval_summary.csv"
    s = pd.read_csv(path)
    assert sorted(s["policy"]) == ["expectimax", "greedy", "random"]
    for col in ("reach_32768", "reach_32768_lo", "reach_32768_hi", "max_tile_32768", "max_tile_65536"):
        assert col in s.columns


def test_compare_reads_both_runs(tmp_path):
    for run, shift in (("a", 3000), ("b", 0)):
        path = V.csv_path(V.Target(run, "final", tmp_path, 1), "greedy", tmp_path)
        path.parent.mkdir(parents=True)
        fake_rows(run, "final", 1, "greedy", 0, 500, shift=shift).to_csv(path, index=False)
    c = RE.compare("a", "b", eval_root=tmp_path)
    assert c["score_diff"] == pytest.approx(3000) and c["rate_diff"] == 0 and c["games"] == 500
    with pytest.raises(FileNotFoundError):
        RE.compare("a", "missing", eval_root=tmp_path)


def test_progression_includes_final_at_its_games_trained(table):
    fig = RE.plot_progression(table)
    for ax in fig.axes:
        final = {l.get_gid(): l for l in ax.get_lines() if (l.get_gid() or "").startswith("final:")}
        assert set(final) == {"final:overnight-a005", "final:overnight-a0025"}
        assert final["final:overnight-a005"].get_xdata().tolist() == [12_900_000]
        assert ax.get_xscale() == "log"


def test_max_tile_distribution_has_32768_and_65536_bins(table):
    fig = RE.plot_max_tile_distribution(table, "overnight-a005")
    ax = fig.axes[0]
    bins = {p.get_gid() for p in ax.patches}
    assert {"bin:32768", "bin:65536", "bin:1024"} <= bins
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["random", "greedy", "expectimax-d1", "expectimax-d2"]
    widths = np.zeros(4)
    for p in ax.patches:
        widths[round(3 - p.get_y() - p.get_height() / 2)] += p.get_width()
    np.testing.assert_allclose(widths, 100)


def test_strength_vs_cost_prefers_bench_timing(table):
    ax = RE.plot_strength_vs_cost(table, "overnight-a005").axes[0]
    assert "parallel" in ax.get_xlabel()
    t = table.assign(ms_per_move_bench=table["ms_per_move"] / 2)
    ax = RE.plot_strength_vs_cost(t, "overnight-a005").axes[0]
    assert "single process" in ax.get_xlabel() and ax.get_xscale() == "log"


@pytest.fixture
def results():
    parts = [fake_rows("-", "-", 0, "random", 0, 50)]
    parts.append(fake_rows("overnight-a005", "final", 12_900_000, "greedy", 0, 300, seed=3))
    for d in (1, 2):
        parts.append(fake_rows("overnight-a005", "final", 12_900_000, "expectimax", d, 200, seed=d,
                               shift=20_000 * d))
    df = pd.concat(parts, ignore_index=True)
    df["max_tile_pos"] = np.random.default_rng(0).choice([0, 3, 12, 15, 5], len(df))
    return df


def test_best_expectimax_and_policy_rows(results):
    t = S.summarize_results(results)
    assert RE.best_expectimax(t, "overnight-a005") == "expectimax-d2"
    assert RE.best_expectimax(t, "nope") is None
    rows = RE.policy_rows(t, "overnight-a005")
    assert [RE.policy_name(p, d) for p, d in zip(rows["policy"], rows["depth"])] == \
        ["random", "greedy", "expectimax-d1", "expectimax-d2"]


@pytest.mark.parametrize("col", ["score", "moves"])
def test_distributions_one_panel_per_policy(results, col):
    fig = RE.plot_distributions(results, "overnight-a005", col)
    assert [ax.get_title(loc="left").split(" ·")[0] for ax in fig.axes] == \
        ["greedy", "expectimax-d1", "expectimax-d2"]
    for ax in fig.axes:
        heights = sum(p.get_height() for p in ax.patches)
        assert heights == pytest.approx(100)  # % of games per panel


def test_max_tile_heatmap_shares(results):
    fig = RE.plot_max_tile_heatmap(results, "overnight-a005", ["greedy", "expectimax-d2"])
    assert len(fig.axes) == 2
    for ax, label in zip(fig.axes, ["greedy", "expectimax-d2"]):
        grid = ax.get_images()[0].get_array()
        assert grid.shape == (4, 4) and grid.sum() == pytest.approx(100)
        pos = RE._games_of(results, "overnight-a005", label)["max_tile_pos"]
        assert grid[0, 0] == pytest.approx(100 * (pos == 0).mean())
    assert "no corner is preferred" in fig._supxlabel.get_text()


def test_corner_frac_figure(results):
    t = S.summarize_results(results)
    ax = RE.plot_corner_frac(t, "overnight-a005").axes[0]
    bars = [p for p in ax.patches if p.get_gid() == "bar:corner_frac"]
    assert len(bars) == 4
    greedy = t[t["policy"] == "greedy"].iloc[0]
    assert sorted(b.get_width() for b in bars)[0] >= 0
    assert any(b.get_width() == pytest.approx(100 * greedy["corner_frac"]) for b in bars)


def test_headline_json(results, tmp_path, ckpt_root):
    import json

    from ntuple import train as T

    eval_root = tmp_path / "results" / "eval"
    for (run, ckpt, policy, depth), g in results.groupby(["run", "checkpoint", "policy", "depth"]):
        label = V.policy_label(policy, depth)
        target = V.RANDOM_TARGET if policy == "random" else V.Target(run, ckpt, tmp_path, 1)
        path = V.csv_path(target, label, eval_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(path, index=False)
    (tmp_path / "ck" / "overnight-a005").mkdir(parents=True)
    (tmp_path / "ck" / "overnight-a005" / "meta.json").write_text(
        json.dumps({"games_done": 12_900_000, "total_moves": 9 * 10**10, "elapsed_s": 28_800}))
    (tmp_path / "logs").mkdir()
    pd.DataFrame({c: [1.0] for c in T.LOG_FIELDS}).assign(alpha=0.05).to_csv(
        tmp_path / "logs" / "overnight-a005.csv", index=False)
    path = RE.write_headline("overnight-a005", eval_root=eval_root, ckpt_root=tmp_path / "ck",
                             log_root=tmp_path / "logs")
    head = json.loads(path.read_text())
    assert path == tmp_path / "results" / "summary.json"
    assert head["best_policy"] == "expectimax-d2" and head["training_hours"] == 8.0
    assert set(head["policies"]) == {"random", "greedy", "expectimax-d1", "expectimax-d2"}
    assert head["count_32768_total"] == sum(int((g["max_tile"] >= 32768).sum()) for _, g in
                                            results.groupby(["policy", "depth"]))
    assert head["best"]["reach"]["32768"]["pct"] >= 0 and head["alpha"] == 0.05
