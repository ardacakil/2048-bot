# Training the n-tuple network

All commands run from the repo root in normal Terminal windows (no tmux). Each training
run is a single process on one core; checkpoints go to `checkpoints/<run>/`, logs to
`logs/<run>.csv`.

## Before you start

- **Plug in the charger and leave the lid open.** `caffeinate -i` stops the Mac from
  idle-sleeping, but closing the lid still puts it to sleep (unless you use clamshell mode
  with an external display). The screen turning off is fine.
- Leave the Terminal windows open. If one gets closed anyway, that run saves a checkpoint
  and exits cleanly; resume it (below).
- Quit other heavy apps. Three runs use three cores and ~1.2 GB of RAM in total.

## Overnight: three learning rates in parallel

Open three Terminal windows (Terminal → **Cmd+N**) and paste one block into each.

**Window 1, α = 0.1**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a01 --alpha 0.1 --games 20000000 --hours 8 --seed 1
```

**Window 2, α = 0.05**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a005 --alpha 0.05 --games 20000000 --hours 8 --seed 1
```

**Window 3, α = 0.025**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a0025 --alpha 0.025 --games 20000000 --hours 8 --seed 1
```

All three use the same seed, so they differ only in α. Each prints a progress line every
1,000 games.

**Why `--games 20000000 --hours 8`:** the 8-hour cap is what stops each run (it saves a
checkpoint first); 20 M games is a ceiling that will not be reached. Expected: about
**9–10 million games per run** (projection below).

### Checkpoints

Saved every **20,000 games or 20 minutes, whichever comes first**, plus always when a run
stops, so a crash loses at most 20 minutes. Only the latest checkpoint is kept (256 MB per
run).

At overnight speeds 20,000 games take only 30–80 seconds, so the game count always
triggers first: each run writes its 256 MB checkpoint several hundred times (≈300–700 GB of
SSD writes for all three runs over 8 h). To save purely every 20 minutes instead, add
`--checkpoint-every 100000000` to each command.

## Stop a run early

Press **Ctrl+C once** in its window. It finishes the current batch of games (under a second),
saves a checkpoint and exits; the prompt returns once the checkpoint is on disk. A second
Ctrl+C aborts *without* saving, so press it only once.

## Resume a run (after Ctrl+C, a closed window, a crash or a reboot)

Run the same thing with `--resume`, in a window of its own, for example:

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a01 --resume --hours 8
```

Replace `overnight-a01` with `overnight-a005` or `overnight-a0025` for the other runs.
Resuming continues from the last checkpoint with the same settings, weights and random
state, bit-identical to a run that was never interrupted (log rows written after that
checkpoint are dropped and regenerated). `--hours` counts from the moment you resume, so
lower it to whatever is left of the night. Add `--games <new total>` to train further than
the original target.

## Check progress in the morning

- Latest log rows of all three runs (one row per 1,000 games):

  ```bash
  cd ~/Desktop/2048-bot
  tail -n 2 logs/overnight-a01.csv logs/overnight-a005.csv logs/overnight-a0025.csv
  ```

  Columns: `games, moves, alpha, avg_score, max_score, avg_moves, reach_2048, reach_4096,
  reach_8192, reach_16384` (percent of games in that 1,000-game window reaching the tile),
  `games_per_s, moves_per_s, elapsed_s`.
- Plots: open `notebooks/02_ntuple.ipynb`, run the first cell (imports), then the two cells
  under **"Check the overnight runs"**: a status line per run, then all three learning curves
  on shared axes (average score against games and against hours, and the 2048 and 8192 rates).
  They work while the runs are still going.
- Evaluate a finished run (one at a time, after training has stopped, since the evaluation
  uses 8 worker processes):

  ```bash
  ~/.local/bin/uv run python -m ntuple.evaluate --run overnight-a01 --games 1000 --depth 0
  ~/.local/bin/uv run python -m ntuple.evaluate --run overnight-a01 --games 200 --depth 2
  ```

## Measurements behind these numbers (M4, 2026-10-23)

| | |
|---|---|
| speed, 1 run alone | 4.1–4.7 M moves/s; ~1,000 games/s at 400k games (games get longer as the agent improves) |
| speed, 3 runs in parallel | 68% of solo each (they compete for memory bandwidth): ~670 games/s, 3.2 M moves/s |
| memory | ~0.4 GB per run (256 MB weights + game buffers) |
| 100k games | ~1 min alone, ~1.4 min with 3 in parallel |
| 8 h, 3 in parallel | ~9 M games per run if the slowdown keeps its current trend (fit games/s ∝ N^-0.30), up to ~18 M if game length levels off |
| update order | online and backward tied per wall-clock second (online ahead by <1%); online is the default (`--update-order backward` is still available) |
