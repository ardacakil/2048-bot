"""Shared fixtures for the evaluation tests (test_evaluate, test_replays, test_report)."""

import numpy as np
import pytest

from ntuple import evaluate as V
from ntuple import model as M
from ntuple import train as T

EVAL_RUN = "runx"


@pytest.fixture(scope="session")
def W_eval():
    """Random weights: short games, but every code path of a real checkpoint."""
    return np.random.default_rng(1).random((M.NUM_TUPLES, M.TABLE_SIZE), dtype=np.float32) - 0.5


@pytest.fixture(scope="session")
def ckpt_root(tmp_path_factory, W_eval):
    """checkpoints/runx (final, 1,000 games) with one milestone at 500 games."""
    root = tmp_path_factory.mktemp("ckpt")
    T.save_checkpoint(root / EVAL_RUN, W_eval, {"games_done": 1000})
    T.save_checkpoint(T.milestone_dir(root / EVAL_RUN, 500), W_eval, {"games_done": 500})
    return root


@pytest.fixture(scope="session")
def eval_root(tmp_path_factory, ckpt_root):
    """results/eval with random (20 games), greedy (20) and expectimax-d1 (8) on runx/final."""
    root = tmp_path_factory.mktemp("results") / "eval"
    final = V.target_from_path(ckpt_root / EVAL_RUN)
    for target, policy, depth, games in ((V.RANDOM_TARGET, "random", 0, 20), (final, "greedy", 0, 20),
                                         (final, "expectimax", 1, 8)):
        out = V.csv_path(target, V.policy_label(policy, depth), root)
        V.evaluate_target(target, policy, depth, games, workers=2, out_csv=out, verbose=False)
    return root
