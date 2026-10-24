"""Server pieces that need no FastAPI: settings, reach-rate config, wire format, slot, imports."""

import asyncio
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from server import build_config as B
from server import wire
from server.config import ConfigError, Settings, parse_delays
from server.slot import ComputeSlot, QueueFull
from ntuple.timing import Move

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "deploy" / "requirements-server.txt"


# ---------------------------------------------------------------- settings


def test_defaults():
    s = Settings.from_env({})
    assert s == Settings()
    assert s.targets == (2048, 4096, 8192, 16384) and s.delays_ms[16384] == 5
    assert (s.depth, s.queue_cap, s.keepalive_s, s.batch_moves) == (2, 10, 15.0, 256)


def test_overrides():
    s = Settings.from_env({"BOT2048_CHECKPOINT": "checkpoints/x", "BOT2048_DEPTH": "3", "BOT2048_QUEUE_CAP": "4",
                           "BOT2048_DELAYS_MS": "2048:30, 8192:10", "BOT2048_LOG_LEVEL": "DEBUG",
                           "BOT2048_CORS_ORIGINS": "http://localhost:5173, http://127.0.0.1:5173",
                           "BOT2048_KEEPALIVE_S": "20", "BOT2048_BATCH_MS": " "})
    assert s.checkpoint == "checkpoints/x" and s.depth == 3 and s.queue_cap == 4
    assert s.delays_ms == {2048: 30, 8192: 10} and s.targets == (2048, 8192)
    assert s.log_level == "debug" and s.cors_origins == ("http://localhost:5173", "http://127.0.0.1:5173")
    assert s.keepalive_s == 20 and s.batch_ms == 50  # blank = default


@pytest.mark.parametrize("name, value", [("DEPTH", "4"), ("DEPTH", "two"), ("QUEUE_CAP", "-1"),
                                         ("DELAYS_MS", "2000:20"), ("DELAYS_MS", "2048"), ("DELAYS_MS", "2048:-5"),
                                         ("KEEPALIVE_S", "120"), ("LOG_LEVEL", "loud"), ("MAX_MOVES", "0")])
def test_invalid_values_name_the_variable(name, value):
    with pytest.raises(ConfigError, match=f"BOT2048_{name}"):
        Settings.from_env({f"BOT2048_{name}": value})


def test_parse_delays():
    assert parse_delays("2048:20,4096:20,8192:10,16384:5") == {2048: 20, 4096: 20, 8192: 10, 16384: 5}


# ---------------------------------------------------------------- reach rates


def test_build_config_from_a_small_csv(tmp_path):
    fields = ["run", "checkpoint", "games_trained", "policy", "depth", "games"] + [
        f"reach_{t}{s}" for t in B.TILES for s in ("", "_lo", "_hi")]
    path = tmp_path / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for policy, depth, games in (("expectimax", 2, 3000), ("greedy", 0, 10000)):
            w.writerow({"run": "r", "checkpoint": "final", "games_trained": 99, "policy": policy, "depth": depth,
                        "games": games, **{f"reach_{t}{s}": 50 + depth for t in B.TILES for s in ("", "_lo", "_hi")}})
        w.writerow({**{k: 1 for k in fields}, "run": "other", "checkpoint": "final", "policy": "greedy", "depth": 0})
    data = B.build(path, run="r")
    assert list(data["policies"]) == ["greedy", "expectimax-d2"] and data["games_trained"] == 99
    assert B.reach_rate(data, 2, 8192) == {"policy": "expectimax-d2", "games": 3000, "pct": 52.0, "lo": 52.0, "hi": 52.0}
    assert B.reach_rate(data, 0, 2048)["policy"] == "greedy"
    assert B.reach_rate(data, 3, 2048) is None and B.reach_rate(None, 2, 2048) is None
    with pytest.raises(SystemExit):
        B.build(path, run="missing")


def test_committed_reach_rates_match_eval_summary():
    assert B.load() == json.loads(json.dumps(B.build()))
    d2 = B.reach_rate(B.load(), 2, 16384)
    assert d2["games"] >= 1000 and 0 < d2["lo"] <= d2["pct"] <= d2["hi"] <= 100


# ---------------------------------------------------------------- wire format


def test_sse_frames_are_single_line_json():
    frame = wire.sse("queued", wire.queued(2, 3, 4.26))
    assert frame == b'event: queued\ndata: {"position":2,"queue_length":3,"eta_s":4.3}\n\n'
    assert wire.KEEPALIVE.startswith(b":") and wire.KEEPALIVE.endswith(b"\n\n")


def test_steps_and_end_builders():
    moves = [Move(3, 0, 14, 1, 0x1000000000000021, 0.0), Move(0, 4, 2, 1, 0x0000000000010132, 0.0)]
    data, score = wire.steps(5, moves, 100)
    assert data == {"first": 5, "steps": [[3, 0, 14, 1, "0x1000000000000021", 100],
                                          [0, 4, 2, 1, "0x0000000000010132", 104]]}
    assert score == 104
    end = wire.end(True, False, 2, 104, 0xB000000000000001, 0.12345, 1.5)
    assert end["status"] == "reached" and end["max_tile"] == 2048 and end["compute_s"] == 0.123
    assert wire.end(False, True, 2, 0, 1, 0, 0)["status"] == "game_over"
    assert wire.end(False, False, 2, 0, 1, 0, 0)["status"] == "max_moves"


def test_parser_handles_split_chunks_comments_and_multiline_data():
    stream = (wire.sse("meta", {"a": 1}) + wire.KEEPALIVE + b"event: x\ndata: [1,\ndata: 2]\n\n"
              + wire.sse("end", {"ok": True})).decode()
    p = wire.Parser()
    out = []
    for i in range(0, len(stream), 7):  # arbitrary chunk boundaries
        out += p.feed(stream[i:i + 7])
    assert out == [("meta", {"a": 1}), ("comment", "ka"), ("x", [1, 2]), ("end", {"ok": True})]


# ---------------------------------------------------------------- slot


def test_slot_order_cap_and_leaving():
    async def go():
        slot = ComputeSlot(cap=2)
        a, b, c = slot.join(2048), slot.join(4096), slot.join(8192)
        assert a.granted.is_set() and not b.granted.is_set()
        assert [slot.position(t) for t in (a, b, c)] == [0, 1, 2]
        assert slot.ahead(c) == [a, b] and slot.ahead(a) == []
        with pytest.raises(QueueFull):
            slot.join(2048)
        changed = asyncio.create_task(slot.wait_change(5))
        await asyncio.sleep(0)
        slot.leave(b)
        assert await changed and slot.position(c) == 1 and slot.position(b) is None
        assert not await slot.wait_change(0.01)
        slot.release(a)
        assert c.granted.is_set() and slot.position(c) == 0
        slot.release(b)  # not running: ignored
        assert slot.running is c
        d = slot.join(2048)
        d.closed = True  # its client left without leave()
        slot.release(c)
        assert slot.running is None and not slot.waiting

    asyncio.run(go())


# ---------------------------------------------------------------- dependencies


def normalize(dist):
    return re.sub(r"[-_.]+", "_", dist.lower())


def requirement_names():
    """Normalized distribution names pinned in deploy/requirements-server.txt."""
    names = set()
    for line in REQUIREMENTS.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==", line.strip())
        if m:
            names.add(normalize(m.group(1)))
    return names


def requirement_modules():
    """Top-level import names provided by those distributions (e.g. opentelemetry-api -> opentelemetry)."""
    import importlib.metadata

    wanted = requirement_names()
    return {top for top, dists in importlib.metadata.packages_distributions().items()
            if any(normalize(d) in wanted for d in dists)}


def test_requirements_file_is_server_only():
    names = requirement_names()
    assert {"numpy", "numba", "llvmlite", "fastapi", "uvicorn", "starlette"} <= names
    assert not names & {"pandas", "matplotlib", "jupyter", "ipykernel", "ipython", "notebook", "pytest"}


SRC = ROOT / "src"
OURS = {"game", "ntuple", "server"}
ENTRY_POINTS = ["server.app", "server.runner", "server.watch", "server.build_config"]
# imported by dependencies only when present, inside try/except, so absent on the VPS without
# harm: numba probes cffi (+ pycparser) for its cffi integration and yaml for .numba_config.yaml
OPTIONAL_PROBES = {"cffi", "_cffi_backend", "pycparser", "yaml"}
# not packages: module objects created by Cython-compiled extensions, and the standard library's
# platform-specific sysconfig data
PSEUDO_MODULE = re.compile(r"^(_cython_\d+_\d+_\d+|cython_runtime|_sysconfigdata_\w*)$")


def module_path(name):
    base = SRC / name.replace(".", "/")
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


# imports inside functions the server never calls, with the reason
STATIC_EXEMPT = {("ntuple.train", "pandas"): "read_log(), used by notebooks only"}


def static_imports(entry_points):
    """Top-level names of every import statement (functions included) in our modules reachable
    from the entry points, minus STATIC_EXEMPT; returns (top-level names, our modules visited)."""
    import ast

    todo, seen, tops = list(entry_points), set(), set()
    while todo:
        mod = todo.pop()
        path = module_path(mod)
        if mod in seen or path is None:
            continue
        seen.add(mod)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                if (mod, name.split(".")[0]) in STATIC_EXEMPT:
                    continue
                tops.add(name.split(".")[0])
                if name.split(".")[0] in OURS:
                    todo.append(name)
    return tops, seen


def test_our_code_imports_only_vps_dependencies():
    tops, visited = static_imports(ENTRY_POINTS)
    allowed = set(sys.stdlib_module_names) | requirement_modules() | OURS
    assert {"server.app", "ntuple.expectimax", "ntuple.timing", "ntuple.player", "game.engine"} <= visited
    assert "analysis" not in tops and not any(m.startswith("analysis") for m in visited)
    assert not tops - allowed, sorted(tops - allowed)


def test_server_imports_stay_within_the_vps_dependencies():
    code = (
        "import json, sys\n"
        "base = {m.split('.')[0] for m in sys.modules}\n"
        f"import {', '.join(ENTRY_POINTS)}\n"
        "import ntuple.expectimax, ntuple.player, ntuple.timing, game.replay\n"  # imported lazily by the runner
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} - base)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    imported = set(json.loads(out.stdout.strip().splitlines()[-1]))
    allowed = set(sys.stdlib_module_names) | requirement_modules() | OURS | OPTIONAL_PROBES
    unexpected = {m for m in imported - allowed if not PSEUDO_MODULE.match(m)}
    assert not unexpected, sorted(unexpected)
    assert not imported & {"pandas", "matplotlib", "IPython", "analysis"}
    assert not OPTIONAL_PROBES & requirement_modules()  # they really are optional for the VPS
