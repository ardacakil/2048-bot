"""Statistics for evaluation results: confidence intervals, per-group summaries, run comparison.

Intervals are 95% by default (z = 1.96, normal approximation; no scipy needed):
  mean_ci         mean ± z·s/√n
  wilson_ci       Wilson score interval for a proportion (well-behaved near 0% and 100%)
  paired_diff_ci  mean ± CI of the per-seed differences a - b over the seeds both have; game i
                  uses the same seed for every policy and checkpoint (common random numbers)
"""

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
REACH_TILES = (2048, 4096, 8192, 16384, 32768)
MAX_TILE_BINS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)  # 1024 = "1024 or less"
GROUP_KEYS = ["run", "checkpoint", "games_trained", "policy", "depth"]


def mean_ci(x, z=Z95):
    """(mean, lo, hi); the interval is NaN for fewer than 2 values."""
    x = np.asarray(x, np.float64)
    m = x.mean() if len(x) else np.nan
    if len(x) < 2:
        return m, np.nan, np.nan
    half = z * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - half, m + half


def wilson_ci(k, n, z=Z95):
    """(k/n, lo, hi) as fractions, Wilson score interval."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def paired_diff_ci(a, b, z=Z95):
    """(mean of a - b, lo, hi, n) over the common index (seeds) of two Series."""
    common = a.index.intersection(b.index)
    m, lo, hi = mean_ci(a[common].to_numpy(np.float64) - b[common].to_numpy(np.float64), z)
    return m, lo, hi, len(common)


def max_tile_bin(tile):
    """Bin of a max tile: 1024 for anything up to 1024, otherwise the tile (capped at 65536)."""
    return int(min(max(tile, MAX_TILE_BINS[0]), MAX_TILE_BINS[-1]))


def bin_name(b):
    return "le1024" if b == MAX_TILE_BINS[0] else str(b)


def summarize(df):
    """Summary of one group of per-game rows (one run/checkpoint/policy/depth)."""
    n = len(df)
    m, lo, hi = mean_ci(df["score"])
    out = {"games": n, "mean_score": m, "mean_lo": lo, "mean_hi": hi,
           "std_score": df["score"].std(), "median_score": df["score"].median(),
           "max_score": df["score"].max()}
    for t in REACH_TILES:
        p, plo, phi = wilson_ci(int((df["max_tile"] >= t).sum()), n)
        out.update({f"reach_{t}": 100 * p, f"reach_{t}_lo": 100 * plo, f"reach_{t}_hi": 100 * phi})
    bins = df["max_tile"].map(max_tile_bin).value_counts()
    for b in MAX_TILE_BINS:
        out[f"max_tile_{bin_name(b)}"] = 100 * bins.get(b, 0) / n
    out["corner_pct"] = 100 * df["max_tile_in_corner"].astype(bool).mean()
    out["mean_moves"] = df["moves"].mean()
    out["ms_per_move"] = df["wall_s"].sum() / max(1, df["moves"].sum()) * 1e3  # parallel, per worker
    out["n_workers"] = int(df["n_workers"].max())
    return out


def summarize_results(results, bench=None):
    """One summary row per (run, checkpoint, games_trained, policy, depth). `bench` (rows from
    evaluate --bench) adds ms_per_move_bench, the single-process mean over its stages."""
    rows = [{**dict(zip(GROUP_KEYS, key)), **summarize(g)}
            for key, g in results.groupby(GROUP_KEYS, sort=False)]
    if not rows:
        return pd.DataFrame(columns=GROUP_KEYS)
    table = pd.DataFrame(rows)
    if bench is not None and len(bench):
        b = (bench.groupby(["run", "checkpoint", "policy", "depth"])
             .apply(lambda g: g["wall_s"].sum() / g["moves"].sum() * 1e3, include_groups=False)
             .rename("ms_per_move_bench").reset_index())
        table = table.merge(b, on=["run", "checkpoint", "policy", "depth"], how="left")
    return table.sort_values(["run", "games_trained", "policy", "depth"]).reset_index(drop=True)


def _says(lo, hi, name_a, name_b):
    return name_a if lo > 0 else name_b if hi < 0 else "tie"


def compare_runs(a, b, name_a="a", name_b="b", tile=16384):
    """Paired comparison of two per-game result sets on their common seeds.

    Reports the score difference (a - b, 95% CI) together with the `tile` reach-rate difference
    (percentage points, paired CI), each one's verdict, and an overall verdict:
      name_a / name_b  the score CI excludes 0 and the rate difference has the same sign
      "tie"            the score CI includes 0 and the two point estimates do not point opposite ways
      "disagree"       score and rate differences have opposite signs: flag it, do not resolve it
    """
    sa, sb = a.set_index("seed"), b.set_index("seed")
    score = paired_diff_ci(sa["score"], sb["score"])
    rate = paired_diff_ci(100.0 * (sa["max_tile"] >= tile), 100.0 * (sb["max_tile"] >= tile))
    score_says = _says(score[1], score[2], name_a, name_b)
    rate_says = _says(rate[1], rate[2], name_a, name_b)
    if np.sign(score[0]) * np.sign(rate[0]) < 0:
        verdict = "disagree"
    else:
        verdict = score_says
    common = sa.index.intersection(sb.index)
    return {
        "a": name_a, "b": name_b, "tile": tile, "games": score[3],
        "mean_a": sa.loc[common, "score"].mean(), "mean_b": sb.loc[common, "score"].mean(),
        "rate_a": 100 * (sa.loc[common, "max_tile"] >= tile).mean(),
        "rate_b": 100 * (sb.loc[common, "max_tile"] >= tile).mean(),
        "score_diff": score[0], "score_lo": score[1], "score_hi": score[2], "score_says": score_says,
        "rate_diff": rate[0], "rate_lo": rate[1], "rate_hi": rate[2], "rate_says": rate_says,
        "verdict": verdict,
    }


def format_comparison(c):
    t = c["tile"]
    verdict = {"disagree": "DISAGREE: score and reach rate point in opposite directions (not resolved)",
               "tie": "TIE: the score CI includes 0"}.get(c["verdict"], f"winner: {c['verdict']}")
    return "\n".join([
        f"{c['a']} vs {c['b']}, {c['games']:,} paired games (same seeds)",
        f"  mean score     {c['mean_a']:>10,.0f} vs {c['mean_b']:>10,.0f}   "
        f"diff {c['score_diff']:+,.0f}  95% CI [{c['score_lo']:+,.0f}, {c['score_hi']:+,.0f}]  -> {c['score_says']}",
        f"  reach {t}  {c['rate_a']:>9.2f}% vs {c['rate_b']:>9.2f}%   "
        f"diff {c['rate_diff']:+.2f} pp  95% CI [{c['rate_lo']:+.2f}, {c['rate_hi']:+.2f}]  -> {c['rate_says']}",
        f"  {verdict}",
    ])
