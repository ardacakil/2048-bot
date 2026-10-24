"""Generate the data-driven parts of README.md from results/, checkpoints/ and logs/.

    uv run python -m analysis.readme            # rewrite the marked blocks of README.md
    uv run python -m analysis.readme --check    # exit 1 if README.md is out of date

Only text between marker comments is rewritten; everything else in README.md is left alone:

    <!-- headline:start --> ... <!-- headline:end -->    one-sentence result
    <!-- training:start --> ... <!-- training:end -->    games/moves/hours per run
    <!-- results:start -->  ... <!-- results:end -->     results tables + run comparison
    <!-- findings:start --> ... <!-- findings:end -->    findings, each with its numbers

Every number in these blocks is read from results/eval_summary.csv (+ the --bench timings),
checkpoint meta.json files or the training logs; none are typed by hand. Policies evaluated on
fewer than PRELIMINARY_GAMES games are marked preliminary.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import curves as C
from analysis import report as RE
from analysis import stats as S
from ntuple import evaluate as V
from ntuple.train import CHECKPOINT_ROOT, LOG_ROOT, ROOT

README = ROOT / "README.md"
MAIN_RUN = "overnight-a0025"
BASELINE_RUN = "overnight-a005"  # the run MAIN_RUN was compared against
BLOCKS = ("headline", "training", "results", "findings")
PRELIMINARY_GAMES = 1000
DEPTH_LABELS = ("greedy", "expectimax-d1", "expectimax-d2", "expectimax-d3")


# ---------------------------------------------------------------- markers


def markers(name):
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def replace_block(text, name, content):
    """Replace what is between the `name` markers with `content`; the markers stay."""
    start, end = markers(name)
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"README needs exactly one {start} and one {end}")
    i, j = text.index(start) + len(start), text.index(end)
    if j < i:
        raise ValueError(f"{end} comes before {start}")
    return text[:i] + "\n" + content.strip("\n") + "\n" + text[j:]


def update(text, blocks):
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    return text


# ---------------------------------------------------------------- formatting


def k(x, digits=1):
    """211474 -> '211.5k'."""
    return f"{x / 1000:,.{digits}f}k"


def num(x):
    """Scores: plain below 10k (1,093), else k (211.5k)."""
    return f"{x:,.0f}" if abs(x) < 10_000 else k(x)


def score_pm(mean, half):
    """'211.5k ± 1.8k' or '1,093 ± 11': the interval in the same units as the mean."""
    return f"{mean:,.0f} ± {half:,.0f}" if mean < 10_000 else f"{k(mean)} ± {k(half)}"


def millions(n):
    return f"{n / 1e6:.1f}M"


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def per_move(ms):
    return V.format_per_move(ms).replace("/move", "")


def alpha_label(run):
    return C.RUN_LABELS.get(run, run)


def policy_cell(label, games):
    """Policy name for tables, marked preliminary below PRELIMINARY_GAMES games."""
    name = label.replace("expectimax-d", "expectimax d")
    return f"{name} *(preliminary)*" if games < PRELIMINARY_GAMES else name


# ---------------------------------------------------------------- data


def load(eval_root=V.EVAL_ROOT, ckpt_root=CHECKPOINT_ROOT, log_root=LOG_ROOT, run=MAIN_RUN):
    table = RE.summary(eval_root)
    bench = RE.load_bench(eval_root)
    bench = bench[bench["checkpoint"] == "final"] if bench is not None else None
    metas = {r: json.loads((Path(ckpt_root) / r / "meta.json").read_text()) for r in C.RUNS}
    logs = C.load_runs(C.RUNS, log_root)
    return {"table": table, "bench": bench, "metas": metas, "logs": logs, "run": run,
            "eval_root": eval_root, "results": RE.load_results(eval_root)}


def bench_ms(d, label, run=None):
    """Single-process ms/move from --bench on `run`'s final checkpoint (default the main run), or
    None. Random is not tied to a checkpoint; its timing comes from the main run's bench.csv."""
    from ntuple.player import parse_label

    b = d["bench"]
    if b is None:
        return None
    policy, depth = parse_label(label)
    run = d["run"] if policy == "random" else (run or d["run"])
    g = b[(b["run"] == run) & (b["policy"] == policy) & (b["depth"] == depth)]
    return g["wall_s"].sum() / g["moves"].sum() * 1e3 if len(g) else None


def row(d, label, run=None):
    from ntuple.player import parse_label

    policy, depth = parse_label(label)
    t = d["table"]
    sel = (t["policy"] == policy) & (t["depth"] == depth)
    if policy != "random":
        sel &= (t["run"] == (run or d["run"])) & (t["checkpoint"] == "final")
    rows = t[sel]
    return None if rows.empty else rows.iloc[0]


def evaluated(d, run=None):
    return [l for l in DEPTH_LABELS if row(d, l, run) is not None]


def best_label(d, min_games=PRELIMINARY_GAMES):
    """Policy with the highest mean score among those with at least `min_games` games."""
    full = [l for l in evaluated(d) if row(d, l)["games"] >= min_games]
    return max(full, key=lambda l: row(d, l)["mean_score"])


def depth_name(label):
    return label.replace("expectimax-d", "expectimax depth ")


# ---------------------------------------------------------------- blocks


def headline_block(d):
    """Best-mean policy with >= PRELIMINARY_GAMES games; better-scoring preliminary ones second."""
    best = best_label(d)
    r = row(d, best)
    ms = bench_ms(d, best)
    out = (f"**With {depth_name(best)} search, the bot averages {k(r['mean_score'])} points and reaches "
           f"16384 in {pct(r['reach_16384'])} of games at {per_move(ms) if ms else 'n/a'} per move** on one "
           f"CPU core ({int(r['games']):,} games).")
    for label in evaluated(d):
        p = row(d, label)
        if p["games"] < PRELIMINARY_GAMES and p["mean_score"] > r["mean_score"]:
            pm = bench_ms(d, label)
            out += (f" {depth_name(label).capitalize()} does better on a preliminary {int(p['games']):,} games: "
                    f"{k(p['mean_score'])} average, 16384 in {pct(p['reach_16384'])}"
                    + (f", at {per_move(pm)} per move." if pm else "."))
    g = row(d, "greedy")
    if best != "greedy":
        out += (f" Without any search (greedy), it averages {k(g['mean_score'])} and reaches 16384 in "
                f"{pct(g['reach_16384'])} of games at {per_move(bench_ms(d, 'greedy'))} per move.")
    return out


def training_block(d):
    lines = ["| Run | α | Games trained | Moves trained | Wall-clock | Games/s (avg) |",
             "|---|---|---:|---:|---:|---:|"]
    for run in C.RUNS:
        m = d["metas"][run]
        hours = m["elapsed_s"] / 3600
        main = " **(main model)**" if run == d["run"] else ""
        lines.append(f"| `{run}`{main} | {d['logs'][run]['alpha'].iloc[-1]:g} | {millions(m['games_done'])} | "
                     f"{m['total_moves'] / 1e9:.1f}B | {hours:.1f} h | {m['games_done'] / m['elapsed_s']:,.0f} |")
    total_games = sum(m["games_done"] for m in d["metas"].values())
    total_moves = sum(m["total_moves"] for m in d["metas"].values())
    lines += ["", f"In total {millions(total_games)} training games and {total_moves / 1e9:.0f}B moves, "
                  f"three runs in parallel (one core each)."]
    return "\n".join(lines)


def _results_row(d, label, run, prefix=""):
    r = row(d, label, run)
    ms = bench_ms(d, label, run)
    reach = " | ".join(pct(r[f"reach_{t}"]) for t in S.REACH_TILES[:-1])
    n32 = int(r["count_32768"])
    r32 = f"{pct(r['reach_32768'], 2)} ({n32})" if n32 else "0"
    run_cell = "–" if label == "random" else alpha_label(run)
    return (f"| {prefix}{policy_cell(label, r['games'])} | {run_cell} | {int(r['games']):,} | "
            f"{score_pm(r['mean_score'], r['mean_hi'] - r['mean_score'])} | {num(r['median_score'])} | "
            f"{num(r['max_score'])} | {reach} | {r32} | {pct(100 * r['corner_frac'], 0)} | "
            f"{per_move(ms) if ms else '–'} |")


def results_block(d):
    lines = ["| Policy | Run | Games | Mean score ± 95% CI | Median | Max | 2048 | 4096 | 8192 | 16384 | "
             "32768 (games) | Corner | Time/move |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(_results_row(d, "random", None))
    for run in C.RUNS:
        if row(d, "greedy", run) is not None:
            lines.append(_results_row(d, "greedy", run))
    for label in DEPTH_LABELS[1:]:
        if row(d, label) is not None:
            lines.append(_results_row(d, label, d["run"]))
    prelim = [l for l in evaluated(d) if row(d, l)["games"] < PRELIMINARY_GAMES]
    lines += ["",
              "Reach columns: % of games whose largest tile reached at least that tile. *Corner*: share of "
              "moves after which the largest tile sits in a corner. *Time/move*: single-process benchmark "
              "(`--bench`) on each row's checkpoint; random is not tied to a checkpoint, its time is from "
              f"the {alpha_label(d['run'])} benchmark (`bench.csv`). The cheap policies' timings come from a "
              "few milliseconds of play, so small differences between them are noise. "
              "Game i uses seed i for every policy (common random numbers). 95% Wilson intervals for every "
              "rate are in [`results/eval_summary.csv`](results/eval_summary.csv)."]
    if prelim:
        lines.append(f"Rows marked *preliminary* have fewer than {PRELIMINARY_GAMES:,} games.")
    # paired comparison
    c = RE.compare(BASELINE_RUN, d["run"], eval_root=d["eval_root"])
    a, b = alpha_label(d["run"]), alpha_label(BASELINE_RUN)
    lines += ["", f"**{a} vs {b}** (greedy, the same {c['games']:,} seeds): {a} scores "
                  f"{k(-c['score_diff'])} more on average (95% CI {k(-c['score_hi'])} to {k(-c['score_lo'])}) "
                  f"and reaches 16384 {-c['rate_diff']:.1f} percentage points more often "
                  f"(95% CI {-c['rate_hi']:.1f} to {-c['rate_lo']:.1f})."
                  + (" The two measures point in opposite directions, so neither run is clearly better."
                     if c["verdict"] == "disagree" else "")]
    # progression
    lines += ["", "**Greedy strength during training** (mean score · % reaching 16384):", "",
              "| Checkpoint | " + " | ".join(alpha_label(r) for r in C.RUNS) + " |",
              "|---|" + "---:|" * len(C.RUNS)]
    t = d["table"]
    g = t[(t["policy"] == "greedy") & (t["run"].isin(C.RUNS))]
    names = sorted(set(g["checkpoint"]) - {"final"}) + ["final"]
    for name in names:
        cells = []
        for run in C.RUNS:
            r = g[(g["run"] == run) & (g["checkpoint"] == name)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            at = f" at {millions(r['games_trained'])}" if name == "final" else ""
            cells.append(f"{k(r['mean_score'])} · {pct(r['reach_16384'])}{at}")
        label = "final" if name == "final" else f"{C.format_count(int(name.split('-')[1]))} games"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    n = int(g["games"].min()) if len(g) else 0
    lines += ["", f"{n:,} games per cell."]
    return "\n".join(lines)


BURST_RATIO = 3  # steepest rise >= this many times the average rise counts as a burst


def _jump(log, window=20, span=500):
    """Smoothed 16384 rate: steepest rise over `span` rows (start games, end games, from %, to %) and
    the run's average rise per `span` rows, over the whole run."""
    s = C.smooth(log["reach_16384"], window).to_numpy()
    rise = s[span:] - s[:-span]
    i = int(np.nanargmax(rise))
    valid = s[~np.isnan(s)]
    average = (valid[-1] - valid[0]) / ((len(valid) - 1) / span)
    return log["games"].iloc[i], log["games"].iloc[i + span], s[i], s[i + span], average


def chain_score(top_exp):
    """Points for building tile 2**top_exp plus one of every smaller tile, all from 2s: building
    2**k from 2s earns (k - 1) * 2**k."""
    return sum((e - 1) << e for e in range(2, top_exp + 1))


def findings_block(d):
    logs = d["logs"]
    out = []
    # 1. breakthrough
    jumps = {r: _jump(logs[r]) for r in C.RUNS}
    curve = C.summary_table(logs)
    ratio = {r: (jumps[r][3] - jumps[r][2]) / jumps[r][4] for r in C.RUNS}
    parts = [f"{alpha_label(r)}: +{jumps[r][3] - jumps[r][2]:.1f} points ({pct(jumps[r][2])} → "
             f"{pct(jumps[r][3])}, {millions(jumps[r][0])}–{millions(jumps[r][1])} games) vs an average of "
             f"+{jumps[r][4]:.1f}, {ratio[r]:.0f}×" for r in C.RUNS]
    calm = [r for r in C.RUNS if ratio[r] < BURST_RATIO]
    intro = ("In every run the 16384 rate (20k-game smoothed) rises in a burst rather than evenly. "
             if not calm else "")
    order = sorted(C.RUNS, key=lambda r: jumps[r][0])
    alphas = [logs[r]["alpha"].iloc[-1] for r in order]
    when = (" The higher the learning rate, the earlier the burst." if alphas == sorted(alphas, reverse=True)
            else "")
    finals = {r: curve.loc[r, "reach_16384"] for r in C.RUNS}
    out.append("- **The 16384 breakthrough.** " + intro
               + "Steepest rise over 500k games vs the run's average rise per 500k games: " + "; ".join(parts)
               + ". The average covers the whole run, including its flat end, so it understates the climb "
                 "itself."
               + "".join(f" {alpha_label(r)} shows no clear burst (steepest rise only {ratio[r]:.1f}× its "
                         "average)." for r in calm)
               + when + " By the end of training the 16384 rate was "
               + ", ".join(f"{pct(finals[r])} ({alpha_label(r)})" for r in C.RUNS)
               + " over the last 20k games.")
    # 2. alpha trade-off
    gains, early = {}, {}
    for r in C.RUNS:
        sm = C.smooth(logs[r]["avg_score"])
        gains[r] = (sm.iloc[-1] - sm.iloc[-1 - 5000], sm.iloc[-1])  # last 5M games (5,000 rows)
        early[r] = sm[logs[r]["games"] == 1_000_000].iloc[0]
    by_alpha = sorted(C.RUNS, key=lambda r: -logs[r]["alpha"].iloc[-1])  # largest alpha first
    text = ("- **Learning-rate trade-off.** At 1M games the average score was "
            + ", ".join(f"{k(early[r])} ({alpha_label(r)})" for r in by_alpha) + "; at the end "
            + ", ".join(f"{k(gains[r][1])}, after gaining {k(gains[r][0])} in the last 5M games "
                        f"({alpha_label(r)})" for r in by_alpha) + ".")
    if (early[by_alpha[0]] == max(early.values()) and gains[by_alpha[0]][1] == min(g[1] for g in gains.values())):
        text += " The largest α learns fastest at first but levels off lowest."
    out.append(text)
    # 3. frozen vs log
    notes = []
    flagged = []
    for r in C.RUNS:
        fr = row(d, "greedy", r)
        tail = logs[r].tail(20)["avg_score"].mean()
        se = fr["std_score"] * np.sqrt(1 / fr["games"] + 1 / 20_000)
        z = (fr["mean_score"] - tail) / se
        notes.append(f"{alpha_label(r)} {k(tail)} → {k(fr['mean_score'])}")
        if abs(z) > 2:
            flagged.append((r, fr["mean_score"] - tail, abs(z)))
    text = ("- **Frozen weights vs training log.** Average score over the last 20k training games vs the "
            "frozen final weights on 10k fresh games: " + "; ".join(notes) + ".")
    if flagged:
        text += " " + " ".join(
            f"{alpha_label(r)} differs by {k(diff)} (about {z:.0f} standard errors), more than noise. "
            "*Hypothesis, not tested:* with online TD updates the weights keep wandering from snapshot to "
            "snapshot, and this particular snapshot is slightly worse than the average over its last 20k "
            "training games." for r, diff, z in flagged)
        ok = [alpha_label(r) for r in C.RUNS if r not in [f[0] for f in flagged]]
        if ok:
            text += f" The others ({', '.join(ok)}) match their logs within noise."
    else:
        text += " All three match their logs within noise."
    out.append(text)
    # 4. 32768 wall
    greedy_n = sum(int(row(d, "greedy", r)["games"]) for r in C.RUNS)
    greedy_32 = sum(int(row(d, "greedy", r)["count_32768"]) for r in C.RUNS)
    deep = [(l, row(d, l)) for l in evaluated(d)[1:]]
    ceiling = logs[d["run"]].tail(1000)["max_score"].median()
    chain = chain_score(14)
    res = d["results"]
    high = res[res["score"] > 400_000]
    high_32 = int((high["max_tile"] >= 32768).sum())
    share = high_32 / len(high) if len(high) else 0.0
    lowest_32 = res.loc[res["max_tile"] >= 32768, "score"].min()
    above = {r: int((logs[r]["max_score"] > 400_000).sum()) for r in C.RUNS}
    text = (f"- **The 32768 wall.** Greedy play reached 32768 in {greedy_32} of {greedy_n:,} evaluation games; "
            + ", ".join(f"{policy_cell(l, r['games'])} in {int(r['count_32768'])} of {int(r['games']):,}"
                        for l, r in deep)
            + ". *Game arithmetic, not a measurement:* building a 2^k tile from 2s earns (k−1)·2^k points, so "
            f"one 16384 plus one each of 8192, 4096, … down to 4 is worth {chain:,} points; every 4 that spawns "
            "instead of being built from two 2s lowers that by 4. The best game of each 1,000-game training "
            f"window typically scores about {k(ceiling, 0)} (median over the last 1M games of "
            f"{alpha_label(d['run'])}), {100 * ceiling / chain:.0f}% of that figure.")
    if len(high):
        text += (f" In the evaluation games, {len(high)} scored above 400k and {high_32} of them "
                 f"({100 * share:.0f}%) had a 32768 tile"
                 + (f"; no game reached 32768 with less than {k(lowest_32, 0)}" if high_32 else "") + ".")
    hedge = "most likely" if share >= 0.8 else "possibly"
    text += (" The training logs do not record tiles above 16384, but going by that evidence the training "
             f"windows containing a game above 400k ({'; '.join(f'{alpha_label(r)}: {above[r]} of {len(logs[r]):,}' for r in C.RUNS)}) "
             f"{hedge} contained a 32768 game.")
    out.append(text)
    # 5. search vs cost
    steps = []
    labels = evaluated(d)
    for prev, cur in zip(labels, labels[1:]):
        a, b = row(d, prev), row(d, cur)
        ta, tb = bench_ms(d, prev), bench_ms(d, cur)
        cost = f", {tb / ta:.0f}× the time per move" if ta and tb else ""
        steps.append(f"{policy_cell(cur, b['games'])} adds {k(b['mean_score'] - a['mean_score'])} points "
                     f"({100 * (b['mean_score'] / a['mean_score'] - 1):+.0f}%) and "
                     f"{b['reach_16384'] - a['reach_16384']:+.1f} points of 16384 rate over "
                     f"{prev.replace('expectimax-d', 'd')}{cost}")
    out.append("- **Search vs cost.** " + "; ".join(steps) + ".")
    # 6. two groups of games
    out.append(_groups_finding(d))
    return "\n".join(out)


def _groups_finding(d):  # finding 6
    """Median vs mean per policy, explained by the share of games reaching 16384 and the mean score of
    games with and without it (per-game CSVs)."""
    def stats(label):
        r = row(d, label)
        games = RE._games_of(d["results"], d["run"], label)
        made = games["max_tile"] >= 16384
        return r, games.loc[made, "score"].mean(), games.loc[~made, "score"].mean()

    deep = [l for l in evaluated(d) if l != "greedy"]
    above = [l for l in deep if row(d, l)["median_score"] > row(d, l)["mean_score"]]
    if not above:
        return ""
    parts = []
    for label in above:
        r, made, died = stats(label)
        parts.append(f"{policy_cell(label, r['games'])}: median {k(r['median_score'])} vs mean "
                     f"{k(r['mean_score'])}; {pct(r['reach_16384'])} of games reach 16384 and average "
                     f"{k(made)}, the other {pct(100 - r['reach_16384'])} average {k(died)}")
    lead = "For every expectimax policy" if above == deep else "For " + ", ".join(above)
    text = (f"- **Median vs mean.** {lead} the median score is above the mean ("
            + "; ".join(parts) + "). Most games build 16384; the minority that die before it score far "
            "less and pull the mean below the median (see the score histogram).")
    g, made, died = stats("greedy")
    if g["median_score"] < g["mean_score"]:
        text += (f" Greedy is the other way round (median {k(g['median_score'])}, mean {k(g['mean_score'])}): "
                 f"only {pct(g['reach_16384'])} of its games reach 16384 (average {k(made)}), so the median "
                 f"game is one that dies before it (those average {k(died)}).")
    return text


def render(d):
    return {"headline": headline_block(d), "training": training_block(d),
            "results": results_block(d), "findings": findings_block(d)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Regenerate the data-driven blocks of README.md.")
    p.add_argument("--readme", type=Path, default=README)
    p.add_argument("--check", action="store_true", help="only report whether README.md is up to date")
    args = p.parse_args(argv)
    text = args.readme.read_text()
    new = update(text, render(load()))
    if args.check:
        print("README.md is up to date" if new == text else "README.md is out of date")
        sys.exit(0 if new == text else 1)
    args.readme.write_text(new)
    print(f"updated {args.readme}" if new != text else f"{args.readme} already up to date")


if __name__ == "__main__":
    main()
