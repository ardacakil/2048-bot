import csv
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from game import engine as E
from ntuple import evaluate as V
from ntuple import expectimax as X
from ntuple import model as M
from ntuple import train as T


@pytest.fixture(scope="module")
def W_random():
    """Random weights (read-only by convention: tests that mutate weights make their own)."""
    return np.random.default_rng(0).random((M.NUM_TUPLES, M.TABLE_SIZE), dtype=np.float32) - 0.5


def random_board(rng, max_exp=13, p_empty=0.4):
    g = rng.integers(1, max_exp + 1, size=(4, 4))
    g[rng.random((4, 4)) < p_empty] = 0
    return E.encode(g.astype(np.int64))


def random_boards(n, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [random_board(rng, **kw) for _ in range(n)]


def grid_symmetries(grid):
    """The 8 rotations/reflections of a 4x4 numpy grid."""
    out = []
    for k in range(4):
        g = np.rot90(grid, k)
        out += [g, np.fliplr(g)]
    return out


def naive_index(grid, cells):
    flat = grid.flatten()
    return sum(int(flat[c]) << (4 * k) for k, c in enumerate(cells))


# ---------------------------------------------------------------- index extraction


def test_tuple_shapes():
    assert M.TUPLES == ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9), (0, 1, 2, 4, 5, 6), (4, 5, 6, 8, 9, 10))
    assert M.NUM_LOOKUPS == 32


def test_tuple_indices_match_naive():
    for board in random_boards(2000, max_exp=15):
        grid = E.decode(board)
        idx = M.tuple_indices(board)
        for t, cells in enumerate(M.TUPLES):
            assert idx[t] == naive_index(grid, cells)
            assert 0 <= idx[t] < M.TABLE_SIZE


def test_features_match_naive_over_all_symmetries():
    for board in random_boards(2000, seed=1, max_exp=15):
        grid = E.decode(board)
        feats = M.features(board)
        for t, cells in enumerate(M.TUPLES):
            naive = sorted(naive_index(g, cells) for g in grid_symmetries(grid))
            assert sorted(feats[t].tolist()) == naive


def test_symmetries_are_the_dihedral_group():
    grid = np.arange(16, dtype=np.int64).reshape(4, 4) % 16  # all cells distinct
    board = E.encode(grid)
    expected = {E.encode(np.ascontiguousarray(g, dtype=np.int64)) for g in grid_symmetries(grid)}
    got = set(M.symmetries(board).tolist())
    assert got == expected and len(got) == 8


def test_mirror_and_flip_are_involutions():
    for board in random_boards(500, seed=2, max_exp=15):
        assert M.mirror(M.mirror(board)) == board
        assert M.flip(M.flip(board)) == board
        assert E.decode(M.mirror(board)).tolist() == np.fliplr(E.decode(board)).tolist()
        assert E.decode(M.flip(board)).tolist() == np.flipud(E.decode(board)).tolist()


# ---------------------------------------------------------------- value / update


def test_value_is_sum_of_features(W_random):
    for board in random_boards(500, seed=3):
        feats = M.features(board)
        expected = sum(float(W_random[t, i]) for t in range(4) for i in feats[t])
        assert M.value(W_random, board) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_value_equal_for_all_8_symmetric_boards(W_random):
    for board in random_boards(500, seed=4):
        vals = [M.value(W_random, b) for b in M.symmetries(board)]
        assert vals == pytest.approx([vals[0]] * 8, rel=1e-12, abs=1e-9)


def test_update_adds_step_to_every_lookup():
    W = M.new_weights()
    board = E.from_values([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [0, 2, 0, 4]])
    feats = M.features(board)
    M.update(W, board, 0.25)
    for t in range(4):
        for i in set(feats[t].tolist()):
            assert W[t, i] == pytest.approx(0.25 * (feats[t] == i).sum())
    assert W.sum() == pytest.approx(0.25 * 32)


def test_td_update_moves_value_toward_target():
    W = M.new_weights()
    board = E.from_values([[2, 4, 8, 16], [32, 64, 128, 256], [0, 0, 0, 0], [0, 0, 2, 0]])
    assert len(set(map(tuple, M.features(board).T.tolist()))) == 8  # no duplicate lookups
    alpha, target = 0.1, 1000.0
    for _ in range(5):
        before = M.value(W, board)
        M.update(W, board, alpha * (target - before) / M.NUM_LOOKUPS)
        after = M.value(W, board)
        assert abs(target - after) == pytest.approx((1 - alpha) * abs(target - before), rel=1e-5)


def test_greedy_move_matches_python_argmax(W_random):
    for board in random_boards(500, seed=5):
        best = None
        for a in range(4):
            after, r = E.move(board, a)
            if after != board:
                v = M.value(W_random, after)
                if best is None or r + v > best[0]:
                    best = (r + v, a, after, r, v)
        a, after, r, v = M.greedy_move(W_random, board)
        if best is None:
            assert a == -1
        else:
            assert (a, after, r) == best[1:4] and v == pytest.approx(best[4])


# ---------------------------------------------------------------- training loop


def ref_train(W, rng, n_games, alpha, backward):
    """Plain-Python TD(0) (both orders), reusing only value/update/move/spawn."""
    step = alpha / M.NUM_LOOKUPS
    for _ in range(n_games):
        board = E.new_game(rng)
        path = []
        while True:
            best = None
            for a in range(4):
                after, r = E.move(board, a)
                if after != board:
                    v = M.value(W, after)
                    if best is None or r + v > best[0]:
                        best = (r + v, after, r, v)
            if best is None:
                break
            _, after, r, v = best
            if not backward and path:
                prev = path[-1][0]
                M.update(W, prev, step * (r + v - M.value(W, prev)))
            path.append((after, r))
            board = E.spawn_tile(after, rng)
        if backward:
            target = 0.0
            for s, r in reversed(path):
                M.update(W, s, step * (target - M.value(W, s)))
                target = r + M.value(W, s)
        elif path:
            prev = path[-1][0]
            M.update(W, prev, step * (0.0 - M.value(W, prev)))


@pytest.mark.parametrize("order", ["online", "backward"])
def test_train_games_matches_python_reference(order):
    W1, W2 = M.new_weights(), M.new_weights()
    r1, r2 = E.make_rng(7), E.make_rng(7)
    n = 3
    stats = [np.zeros(n, np.int64) for _ in range(3)]
    T.train_games(W1, r1, 0.1, order == "backward", *stats)
    ref_train(W2, r2, n, 0.1, order == "backward")
    assert r1[0] == r2[0]
    assert np.array_equal(W1, W2)
    assert (stats[0] > 0).all() and (stats[2] > 0).all()


def test_backward_and_online_differ_but_both_learn():
    Ws = {}
    for order in ("online", "backward"):
        W = M.new_weights()
        n = 300
        scores = np.zeros(n, np.int64)
        T.train_games(W, E.make_rng(1), 0.1, order == "backward", scores,
                      np.zeros(n, np.int64), np.zeros(n, np.int64))
        Ws[order] = W
        assert scores[-100:].mean() > 2 * 1100  # random play averages ~1,100
    assert not np.array_equal(Ws["online"], Ws["backward"])


# ---------------------------------------------------------------- checkpoints and resume


def small_config(run, games, **kw):
    return T.Config(run=run, games=games, log_every=100, checkpoint_every=50, **kw)


def roots(tmp_path):
    return {"checkpoint_root": tmp_path / "ckpt", "log_root": tmp_path / "logs", "verbose": False}


def read_rows(path, drop_timing=True):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if drop_timing:
        for r in rows:
            for k in T.TIMING_FIELDS:
                r.pop(k)
    return rows


def test_checkpoint_round_trip(tmp_path):
    W = np.random.default_rng(0).random((M.NUM_TUPLES, M.TABLE_SIZE), dtype=np.float32)
    rng = E.make_rng(99)
    meta = {"config": {"run": "x"}, "games_done": 1234, "total_moves": 5678,
            "rng_state": int(rng[0]), "elapsed_s": 1.5, "window": {"scores": [1, 2]}}
    T.save_checkpoint(tmp_path, W, meta)
    for mmap in (False, True):
        W2, meta2 = T.load_checkpoint(tmp_path, mmap=mmap)
        assert np.array_equal(W, W2) and W2.dtype == np.float32 and W2.flags.c_contiguous
        assert {k: meta2[k] for k in meta} == meta
    restored = np.array([meta2["rng_state"]], dtype=np.uint64)
    assert [E.rng_next(restored) for _ in range(5)] == [E.rng_next(rng) for _ in range(5)]


def test_checkpoint_mmap_is_copy_on_write(tmp_path):
    W = M.new_weights()
    T.save_checkpoint(tmp_path, W, {"games_done": 0})
    W2, _ = T.load_checkpoint(tmp_path, mmap=True)
    M.update(W2, 12345, 1.0)  # writable for Numba, but never touches the file
    W3, _ = T.load_checkpoint(tmp_path)
    assert W3.sum() == 0.0 and W2.sum() == 32.0


def test_checkpoint_keeps_only_latest_weights_unless_asked(tmp_path):
    W = M.new_weights()
    for g in (100, 200, 300):
        T.save_checkpoint(tmp_path, W, {"games_done": g}, keep_every=200)
    names = sorted(p.name for p in tmp_path.glob("weights-*.npy"))
    assert names == ["weights-0000000200.npy", "weights-0000000300.npy"]
    assert json.loads((tmp_path / "meta.json").read_text())["weights_file"] == "weights-0000000300.npy"


@pytest.mark.parametrize("order, split", [("online", 150), ("backward", 150), ("backward", 170)])
def test_resume_matches_uninterrupted(tmp_path, order, split):
    full = T.train(small_config("full", 300, update_order=order), **roots(tmp_path))
    # 150 is mid log-window (log_every=100): the pending window must survive the restart;
    # 170 is on no log/checkpoint boundary at all, like a time-triggered checkpoint
    T.train(small_config("split", split, update_order=order), **roots(tmp_path))
    resumed = T.train(resume_run="split", games=300, **roots(tmp_path))
    assert full["games_done"] == resumed["games_done"] == 300
    assert full["total_moves"] == resumed["total_moves"]

    Wa, ma = T.load_checkpoint(tmp_path / "ckpt" / "full")
    Wb, mb = T.load_checkpoint(tmp_path / "ckpt" / "split")
    assert np.array_equal(Wa, Wb)
    assert ma["rng_state"] == mb["rng_state"]
    rows_a = read_rows(tmp_path / "logs" / "full.csv")
    rows_b = read_rows(tmp_path / "logs" / "split.csv")
    assert len(rows_a) == 3 and rows_a == rows_b


@pytest.mark.parametrize("minutes, expected", [
    (1e-9, [100, 200, 300, 400, 500, 500]),  # deadline passed after every 100-game chunk + final
    (0, [500]),  # time-based off, checkpoint_every never reached: only the final save
    (60, [500]),  # deadline never reached
])
def test_time_based_checkpoints(tmp_path, monkeypatch, minutes, expected):
    saves = []
    real = T.save_checkpoint

    def spy(ckpt_dir, W, meta, keep_every=0):
        saves.append(meta["games_done"])
        real(ckpt_dir, W, meta, keep_every)

    monkeypatch.setattr(T, "save_checkpoint", spy)
    cfg = T.Config(run="t", games=500, log_every=100, checkpoint_every=10**9, checkpoint_minutes=minutes)
    T.train(cfg, **roots(tmp_path))
    assert saves == expected


def test_checkpoint_by_games_or_time_whichever_first(tmp_path, monkeypatch):
    saves = []
    real = T.save_checkpoint
    monkeypatch.setattr(T, "save_checkpoint",
                        lambda d, W, meta, k=0: (saves.append(meta["games_done"]), real(d, W, meta, k)))
    cfg = T.Config(run="w", games=500, log_every=100, checkpoint_every=200, checkpoint_minutes=60)
    T.train(cfg, **roots(tmp_path))
    assert saves == [200, 400, 500]


def test_resume_drops_log_rows_after_checkpoint(tmp_path):
    T.train(small_config("r", 200), **roots(tmp_path))
    log = tmp_path / "logs" / "r.csv"
    meta_path = tmp_path / "ckpt" / "r" / "meta.json"
    # simulate a crash after a log row was written but before its checkpoint
    meta = json.loads(meta_path.read_text())
    assert meta["games_done"] == 200
    with open(log, "a") as f:
        f.write("300," + ",".join(["0"] * (len(T.LOG_FIELDS) - 1)) + "\n")
    T.train(resume_run="r", games=200, **roots(tmp_path))
    assert [r["games"] for r in read_rows(log)] == ["100", "200"]


def test_new_run_refuses_to_overwrite(tmp_path):
    T.train(small_config("dup", 50), **roots(tmp_path))
    with pytest.raises(FileExistsError):
        T.train(small_config("dup", 50), **roots(tmp_path))


def test_alpha_decay_schedule(tmp_path):
    cfg = small_config("decay", 300, alpha=0.2, decay_factor=0.5, decay_every=150)
    assert [cfg.alpha_at(g) for g in (0, 149, 150, 299, 300)] == [0.2, 0.2, 0.1, 0.1, 0.05]
    assert T.Config(run="c", games=1).alpha_at(10**9) == 0.1
    T.train(cfg, **roots(tmp_path))
    rows = read_rows(tmp_path / "logs" / "decay.csv")
    assert [float(r["alpha"]) for r in rows] == [0.2, 0.1, 0.1]


def test_config_validation():
    with pytest.raises(ValueError):
        T.Config(run="x", games=10, update_order="sideways")
    with pytest.raises(ValueError):
        T.Config(run="x", games=0)


def test_log_contents(tmp_path):
    T.train(small_config("log", 200), **roots(tmp_path))
    rows = read_rows(tmp_path / "logs" / "log.csv", drop_timing=False)
    assert list(rows[0]) == T.LOG_FIELDS
    assert [int(r["games"]) for r in rows] == [100, 200]
    for r in rows:
        assert float(r["avg_score"]) > 0 and float(r["games_per_s"]) > 0 and float(r["moves_per_s"]) > 0
        assert 0 <= float(r["reach_2048"]) <= 100


def test_sigint_saves_checkpoint_and_resumes(tmp_path):
    ckpt, logs = tmp_path / "ckpt", tmp_path / "logs"
    cmd = [sys.executable, "-m", "ntuple.train", "--run", "sig", "--games", "100000000",
           "--log-every", "100", "--checkpoint-every", "1000000",
           "--checkpoint-root", str(ckpt), "--log-root", str(logs)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = logs / "sig.csv"
    deadline = time.time() + 120
    while time.time() < deadline and (not log.exists() or len(log.read_text().splitlines()) < 3):
        time.sleep(0.1)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert "interrupted" in out
    meta = json.loads((ckpt / "sig" / "meta.json").read_text())
    done = meta["games_done"]
    assert done >= 200 and done % T.CHUNK_GAMES == 0
    summary = T.train(resume_run="sig", games=done + 100, **roots(tmp_path))
    assert summary["games_done"] == done + 100


def test_closing_terminal_window_saves_checkpoint(tmp_path):
    """Run training on a pseudo-terminal, then close it: SIGHUP + a dead stdout."""
    import pty
    import select

    ckpt, logs = tmp_path / "ckpt", tmp_path / "logs"
    cmd = [sys.executable, "-m", "ntuple.train", "--run", "hup", "--games", "100000000",
           "--log-every", "100", "--checkpoint-every", "1000000",
           "--checkpoint-root", str(ckpt), "--log-root", str(logs)]
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, cmd)
    log = logs / "hup.csv"
    deadline = time.time() + 120
    while time.time() < deadline and (not log.exists() or len(log.read_text().splitlines()) < 3):
        if select.select([fd], [], [], 0.1)[0]:
            os.read(fd, 65536)  # keep the terminal drained like a real window would
    os.close(fd)  # window closed: the kernel hangs up the terminal and sends SIGHUP
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
    meta = json.loads((ckpt / "hup" / "meta.json").read_text())
    assert meta["games_done"] >= 200
    assert int(read_rows(log)[-1]["games"]) <= meta["games_done"]


# ---------------------------------------------------------------- expectimax


def ref_chance(W, after, depth):
    """Plain-Python exact expectimax (no pruning, no table)."""
    if depth == 0:
        return M.value(W, after)
    empties = [i for i in range(16) if (after >> (4 * i)) & 0xF == 0]
    total = 0.0
    for i in empties:
        for tile, p in ((1, 0.9), (2, 0.1)):
            child = after | (tile << (4 * i))
            vals = [r + ref_chance(W, a2, depth - 1)
                    for a in range(4) for a2, r in [E.move(child, a)] if a2 != child]
            total += p * (max(vals) if vals else 0.0)
    return total / len(empties)


def ref_best(W, board, depth):
    best = None
    for a in range(4):
        after, r = E.move(board, a)
        if after != board:
            v = r + ref_chance(W, after, depth)
            if best is None or v > best[1]:
                best = (a, v)
    return best or (-1, 0.0)


def playable_boards(n, seed, **kw):
    return [b for b in random_boards(4 * n, seed=seed, **kw) if not E.is_game_over(b)][:n]


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_expectimax_returns_legal_move(W_random, depth):
    ex = X.Expectimax(W_random, depth=depth)
    for board in playable_boards(150 if depth < 2 else 40, seed=10 + depth, p_empty=0.5):
        a, _ = ex.best_move(board)
        assert 0 <= a < 4 and E.is_legal(board, a)


def test_expectimax_no_move_on_finished_board(W_random):
    board = E.from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    for depth in (0, 2):
        assert X.Expectimax(W_random, depth=depth).best_move(board)[0] == -1


@pytest.mark.parametrize("weights", ["random", "zero"])
def test_depth0_expectimax_equals_greedy(W_random, weights):
    W = W_random if weights == "random" else M.new_weights()  # zero weights: many ties
    ex = X.Expectimax(W, depth=0)
    for board in playable_boards(1000, seed=20):
        a, v = ex.best_move(board)
        ga, after, r, gv = M.greedy_move(W, board)
        assert a == ga
        assert v == pytest.approx(r + gv)


@pytest.mark.parametrize("depth, p_empty", [(1, 0.5), (2, 0.25)])
def test_expectimax_matches_python_reference(W_random, depth, p_empty):
    ex = X.Expectimax(W_random, depth=depth, prune=0.0, use_tt=False)
    boards = [b for b in playable_boards(200, seed=30 + depth, p_empty=p_empty)
              if E.count_empty(b) <= 6][:25]
    assert len(boards) >= 10
    for board in boards:
        a, v = ex.best_move(board)
        ra, rv = ref_best(W_random, board, depth)
        assert a == ra and v == pytest.approx(rv, rel=1e-9)


def test_transposition_table_is_exact_without_pruning(W_random):
    plain = X.Expectimax(W_random, depth=2, prune=0.0, use_tt=False)
    cached = X.Expectimax(W_random, depth=2, prune=0.0, tt_bits=12)  # small: forces collisions
    for board in playable_boards(60, seed=40, p_empty=0.5):
        assert cached.best_move(board) == pytest.approx(plain.best_move(board), rel=1e-12)
    assert (cached.table[1] >= 0).sum() > 0  # the table was actually used


def test_pruning_everything_below_root_equals_depth_1(W_random):
    # children of the root chance layer have reach probability < 1, so prune=1.0 stops there
    pruned = X.Expectimax(W_random, depth=3, prune=1.0, use_tt=False)
    depth1 = X.Expectimax(W_random, depth=1, prune=0.0, use_tt=False)
    for board in playable_boards(100, seed=50):
        assert pruned.best_move(board) == depth1.best_move(board)


def test_expectimax_play_game_reproducible(W_random):
    a = X.Expectimax(W_random, depth=1).play_game(E.make_rng(5))
    b = X.Expectimax(W_random, depth=1).play_game(E.make_rng(5))
    assert a == b and a[2] > 0


# ---------------------------------------------------------------- evaluation


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    T.train(T.Config(run="ev", games=300, checkpoint_every=300, log_every=100), **roots(tmp))
    return tmp / "ckpt" / "ev"


def test_evaluate_reproducible_across_worker_counts(trained_checkpoint):
    s1, df1 = V.evaluate(trained_checkpoint, games=6, depth=1, workers=1, verbose=False)
    s3, df3 = V.evaluate(trained_checkpoint, games=6, depth=1, workers=3, verbose=False)
    assert df1[["game", "score", "moves", "max_tile"]].equals(df3[["game", "score", "moves", "max_tile"]])
    assert s1["avg_score"] == s3["avg_score"] > 1100
    assert sum(s1["max_tile_pct"].values()) == pytest.approx(100)
    assert s1["ms_per_move"] > 0


# ---------------------------------------------------------------- uint64 typing

# same IR check as tests/test_engine.py: no sign-extending shifts anywhere; pure-integer
# helpers additionally must not convert between int and float
_IR_CHECK = """
import json, re, sys
from game import engine
from ntuple import model, train, expectimax
mods = {"model": model, "train": train, "expectimax": expectimax}
OPS = re.compile(r"\\b(uitofp|sitofp|fptoui|fptosi|ashr)\\b")
out = {}
for spec in sys.argv[1:]:
    mod, name = spec.split(":")
    fn = getattr(mods[mod], name)
    out[spec] = sorted(set(OPS.findall(fn.inspect_llvm(fn.signatures[0]))))
print(json.dumps(out))
"""
INT_ONLY = ["model:mirror", "model:flip", "model:symmetries", "model:tuple_indices",
            "model:features", "expectimax:_slot"]
WITH_FLOATS = ["model:value", "model:update", "model:greedy_move", "train:train_games",
               "expectimax:chance_value", "expectimax:best_move", "expectimax:play_game"]


def test_no_signed_or_float_promotion_in_compiled_code(tmp_path):
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-c", _IR_CHECK, *INT_ONLY, *WITH_FLOATS],
                          env=env, capture_output=True, text=True, check=True)
    found = json.loads(proc.stdout.strip().splitlines()[-1])
    # liveness: reward (int) + V (float) must show up, else inspection returned nothing useful
    assert set(found["model:greedy_move"]) & {"uitofp", "sitofp"}
    for spec in INT_ONLY:
        assert found[spec] == [], spec
    for spec in WITH_FLOATS:
        assert "ashr" not in found[spec], spec
