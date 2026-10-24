"""Build server/reach_rates.json from results/eval_summary.csv (run on the Mac, commit the JSON).

    uv run python -m server.build_config [--run overnight-a0025] [--summary results/eval_summary.csv]

The VPS has no evaluation CSVs; the server reads this JSON to tell viewers how often its policy
reaches each target. Standard library only (csv), so it runs anywhere.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "eval_summary.csv"
OUT = Path(__file__).resolve().with_name("reach_rates.json")
TILES = (2048, 4096, 8192, 16384, 32768)


def label(policy, depth):
    return f"expectimax-d{depth}" if policy == "expectimax" else policy


def build(summary=SUMMARY, run="overnight-a0025", checkpoint="final"):
    """{"run", "checkpoint", "games_trained", "policies": {label: {"depth", "games", "targets"}}}."""
    with open(summary, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["run"] == run and r["checkpoint"] == checkpoint]
    if not rows:
        raise SystemExit(f"no rows for {run}/{checkpoint} in {summary}")
    policies = {}
    for r in sorted(rows, key=lambda r: (r["policy"] != "greedy", int(r["depth"]))):
        policies[label(r["policy"], int(r["depth"]))] = {
            "depth": int(r["depth"]),
            "games": int(r["games"]),
            "targets": {str(t): {"pct": round(float(r[f"reach_{t}"]), 2),
                                 "lo": round(float(r[f"reach_{t}_lo"]), 2),
                                 "hi": round(float(r[f"reach_{t}_hi"]), 2)} for t in TILES},
        }
    return {"source": "results/eval_summary.csv", "run": run, "checkpoint": checkpoint,
            "games_trained": int(rows[0]["games_trained"]), "policies": policies}


def write(data, path=OUT):
    Path(path).write_text(json.dumps(data, indent=1) + "\n")
    return path


def load(path=OUT):
    return json.loads(Path(path).read_text())


def reach_rate(data, depth, target):
    """{"policy", "games", "pct", "lo", "hi"} for this depth and target, or None if not evaluated."""
    name = label("expectimax" if depth else "greedy", depth)
    p = (data or {}).get("policies", {}).get(name)
    if not p or str(target) not in p["targets"]:
        return None
    return {"policy": name, "games": p["games"], **p["targets"][str(target)]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", type=Path, default=SUMMARY)
    p.add_argument("--run", default="overnight-a0025")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    print(f"wrote {write(build(args.summary, args.run), args.out)}")


if __name__ == "__main__":
    main()
