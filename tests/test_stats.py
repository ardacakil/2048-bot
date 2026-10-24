import numpy as np
import pandas as pd
import pytest

from analysis import stats as S


def rows(scores, tiles, seeds=None, corner=True, **fixed):
    n = len(scores)
    base = {"run": "r", "checkpoint": "final", "games_trained": 1000, "policy": "greedy", "depth": 0}
    return pd.DataFrame({**base, **fixed, "seed": seeds if seeds is not None else np.arange(n),
                         "score": scores, "max_tile": tiles, "max_tile_in_corner": corner,
                         "moves": 100, "wall_s": 0.5, "n_workers": 8})


def test_mean_ci():
    m, lo, hi = S.mean_ci([1, 2, 3, 4, 5])
    half = 1.959963984540054 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
    assert m == 3 and lo == pytest.approx(3 - half) and hi == pytest.approx(3 + half)
    assert np.isnan(S.mean_ci([7])[1])


@pytest.mark.parametrize("k, n, lo, hi", [(5, 10, 0.2366, 0.7634), (0, 10, 0.0, 0.2775),
                                          (10, 10, 0.7225, 1.0), (81, 263, 0.2553, 0.3662)])
def test_wilson_ci_textbook_values(k, n, lo, hi):
    p, a, b = S.wilson_ci(k, n)
    assert p == k / n and a == pytest.approx(lo, abs=1e-4) and b == pytest.approx(hi, abs=1e-4)


def test_wilson_ci_empty():
    assert all(np.isnan(S.wilson_ci(0, 0)))


def test_paired_diff_ci_uses_common_seeds_only():
    a = pd.Series([10.0, 20.0, 30.0, 99.0], index=[0, 1, 2, 3])
    b = pd.Series([8.0, 19.0, 27.0, 5.0], index=[0, 1, 2, 7])
    m, lo, hi, n = S.paired_diff_ci(a, b)
    assert n == 3 and m == pytest.approx(2.0)
    assert (m, lo, hi) == pytest.approx(S.mean_ci([2.0, 1.0, 3.0]))


def test_max_tile_bins():
    assert [S.max_tile_bin(t) for t in (2, 512, 1024, 2048, 32768, 65536, 131072)] == \
        [1024, 1024, 1024, 2048, 32768, 65536, 65536]
    assert S.bin_name(1024) == "le1024" and S.bin_name(65536) == "65536"


def test_summarize():
    tiles = [512, 2048, 4096, 8192, 16384, 16384, 32768, 65536]
    s = S.summarize(rows([100, 200, 300, 400, 500, 600, 700, 800], tiles,
                         corner=[True] * 6 + [False] * 2))
    assert s["games"] == 8 and s["mean_score"] == 450 and s["max_score"] == 800
    assert s["reach_2048"] == pytest.approx(7 / 8 * 100)
    assert s["reach_16384"] == 50 and s["reach_32768"] == 25
    p, lo, hi = S.wilson_ci(2, 8)
    assert (s["reach_32768_lo"], s["reach_32768_hi"]) == pytest.approx((100 * lo, 100 * hi))
    assert s["max_tile_le1024"] == 12.5 and s["max_tile_16384"] == 25
    assert s["max_tile_32768"] == 12.5 and s["max_tile_65536"] == 12.5
    assert sum(s[f"max_tile_{S.bin_name(b)}"] for b in S.MAX_TILE_BINS) == pytest.approx(100)
    assert s["corner_pct"] == 75 and s["ms_per_move"] == pytest.approx(5.0)


def test_summarize_results_groups_and_merges_bench():
    df = pd.concat([rows([1, 2, 3], [2048] * 3), rows([4, 5], [4096] * 2, policy="expectimax", depth=1),
                    rows([9, 9], [8192] * 2, checkpoint="games-500", games_trained=500)])
    bench = pd.DataFrame({"run": "r", "checkpoint": "final", "policy": ["greedy", "greedy"], "depth": 0,
                          "moves": [100, 300], "wall_s": [0.1, 0.1]})
    t = S.summarize_results(df, bench)
    assert len(t) == 3
    assert t["games_trained"].tolist() == [500, 1000, 1000]  # sorted by games trained
    g = t[(t["policy"] == "greedy") & (t["checkpoint"] == "final")].iloc[0]
    assert g["games"] == 3 and g["ms_per_move_bench"] == pytest.approx(0.5)
    assert t[t["policy"] == "expectimax"]["ms_per_move_bench"].isna().all()
    assert "reach_32768_hi" in t.columns and "max_tile_65536" in t.columns
    assert S.summarize_results(df.iloc[:0]).empty


def comparison(score_a, score_b, tiles_a, tiles_b):
    return S.compare_runs(rows(score_a, tiles_a), rows(score_b, tiles_b), "A", "B")


def test_compare_runs_winner_tie_and_disagreement():
    rng = np.random.default_rng(0)
    base = rng.normal(200_000, 50_000, 2000)
    big, small = [16384] * 1200 + [8192] * 800, [16384] * 800 + [8192] * 1200
    win = comparison(base + 5000, base + rng.normal(0, 1000, 2000), big, small)
    assert win["verdict"] == "A" and win["score_says"] == "A" and win["rate_says"] == "A"
    assert win["rate_diff"] == pytest.approx(20.0) and win["games"] == 2000

    tie = comparison(base + rng.normal(0, 20_000, 2000), base + rng.normal(0, 20_000, 2000), big, big)
    assert tie["verdict"] == "tie" and tie["score_lo"] < 0 < tie["score_hi"]

    flag = comparison(base + 5000, base, small, big)  # higher score, lower 16384 rate
    assert flag["verdict"] == "disagree" and flag["score_says"] == "A" and flag["rate_says"] == "B"
    assert "DISAGREE" in S.format_comparison(flag) and "winner: A" in S.format_comparison(win)

    weak = comparison(base + rng.normal(0, 20_000, 2000), base + rng.normal(0, 20_000, 2000),
                      [16384] * 1001 + [8192] * 999, big)
    assert weak["verdict"] in ("tie", "disagree")  # never a winner without a significant score gap
