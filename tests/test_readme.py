import pytest

from analysis import readme as RM

TEXT = """# Title

intro stays
<!-- results:start -->
old table
<!-- results:end -->
middle stays
<!-- headline:start --><!-- headline:end -->
tail stays
"""


def test_replace_block_rewrites_only_between_markers():
    new = RM.replace_block(TEXT, "results", "| a | b |\n|---|---|\n")
    assert "old table" not in new and "| a | b |" in new
    before, after = TEXT.split("<!-- results:start -->"), new.split("<!-- results:start -->")
    assert before[0] == after[0]  # everything above is untouched
    assert new.split("<!-- results:end -->")[1] == TEXT.split("<!-- results:end -->")[1]
    assert "<!-- results:start -->\n| a | b |\n|---|---|\n<!-- results:end -->" in new


def test_replace_block_is_idempotent_and_handles_empty_blocks():
    once = RM.update(TEXT, {"results": "x", "headline": "**y**"})
    assert RM.update(once, {"results": "x", "headline": "**y**"}) == once
    assert "<!-- headline:start -->\n**y**\n<!-- headline:end -->" in once
    assert once.count("stays") == 3


@pytest.mark.parametrize("text", [
    "no markers at all",
    "<!-- results:start --> only a start",
    "<!-- results:end --> backwards <!-- results:start -->",
    "<!-- results:start --><!-- results:end --><!-- results:start --><!-- results:end -->",
])
def test_replace_block_rejects_bad_markers(text):
    with pytest.raises(ValueError):
        RM.replace_block(text, "results", "x")


def test_formatting_helpers():
    assert RM.k(211474) == "211.5k" and RM.k(5477) == "5.5k" and RM.k(384000, 0) == "384k"
    assert RM.millions(13_936_900) == "13.9M"
    assert RM.pct(48.3) == "48.3%" and RM.pct(0.37, 2) == "0.37%"
    assert RM.per_move(0.00022) == "0.22 µs" and RM.per_move(1.874) == "1.874 ms"


def test_small_scores_are_not_rounded_away():
    assert RM.score_pm(1093.2, 10.6) == "1,093 ± 11"
    assert RM.score_pm(211474, 1791) == "211.5k ± 1.8k"
    assert RM.num(5312) == "5,312" and RM.num(387300) == "387.3k"


def test_policy_cell_marks_preliminary_rows():
    assert RM.policy_cell("expectimax-d3", 999) == "expectimax d3 *(preliminary)*"
    assert RM.policy_cell("expectimax-d3", 1000) == "expectimax d3"
    assert RM.policy_cell("greedy", 10_000) == "greedy"


def test_readme_has_every_block():
    text = RM.README.read_text()
    for name in RM.BLOCKS:
        start, end = RM.markers(name)
        assert text.count(start) == 1 and text.count(end) == 1, name


def test_chain_score_is_game_arithmetic():
    # building 2**k from 2s earns (k - 1) * 2**k: 4 -> 4, 8 -> 4 + 2*8 = 20, ...
    assert RM.chain_score(2) == 4 and RM.chain_score(3) == 4 + 16
    assert RM.chain_score(11) == 36_868
    assert RM.chain_score(14) == sum((k - 1) * 2**k for k in range(2, 15)) == 393_220


def test_headline_policy_needs_enough_games():
    import pandas as pd

    t = pd.DataFrame({"policy": ["greedy", "expectimax", "expectimax"], "depth": [0, 2, 3],
                      "run": "r", "checkpoint": "final", "games": [10_000, 3_000, 300],
                      "mean_score": [211_000.0, 320_000.0, 333_000.0]})
    d = {"table": t, "run": "r"}
    assert RM.best_label(d) == "expectimax-d2"  # d3 has fewer than PRELIMINARY_GAMES
    assert RM.best_label(d, min_games=100) == "expectimax-d3"
    t.loc[2, "games"] = 1_000  # a finished overnight d3 run takes over
    assert RM.best_label(d) == "expectimax-d3"
