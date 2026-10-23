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
- Free disk: about 5 GB for checkpoints and milestones (62 GB free on 2026-10-24).

## Overnight: three learning rates in parallel

Open three Terminal windows (Terminal → **Cmd+N**) and paste one block into each.

**Window 1, α = 0.1**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a01 --alpha 0.1 --games 20000000 --hours 8 --seed 1 --checkpoint-every 100000000 --milestones 1000,10000,100000,1000000,5000000
```

**Window 2, α = 0.05**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a005 --alpha 0.05 --games 20000000 --hours 8 --seed 1 --checkpoint-every 100000000 --milestones 1000,10000,100000,1000000,5000000
```

**Window 3, α = 0.025**

```bash
cd ~/Desktop/2048-bot
caffeinate -i ~/.local/bin/uv run python -m ntuple.train --run overnight-a0025 --alpha 0.025 --games 20000000 --hours 8 --seed 1 --checkpoint-every 100000000 --milestones 1000,10000,100000,1000000,5000000
```

All three use the same seed, so they differ only in α. Each prints a progress line every
1,000 games.

**Why `--games 20000000 --hours 8`:** the 8-hour cap is what stops each run (it saves a
checkpoint first); 20 M games is a ceiling that will not be reached. Expected: about
**9–10 million games per run** (projection below).

### Checkpoints

The commands save a checkpoint **every 20 minutes**, plus always when a run stops, so a
crash loses at most 20 minutes. Only the latest checkpoint is kept: each save deletes the
previous weights file, so each run uses 256 MB for its checkpoint however long it trains.

`--checkpoint-every 100000000` is what makes it purely time-based. Without it the default is
every 20,000 games *or* 20 minutes, and at overnight speeds 20,000 games take only
30–80 seconds, so each run would rewrite its 256 MB checkpoint several hundred times
(≈300–700 GB of SSD writes for all three runs over 8 h). With it: ~24 saves per run, ≈18 GB
of writes in total.

If the Mac crashes or loses power *while* a checkpoint is being written, the previous
checkpoint stays valid. The half-written `weights-*.npy.tmp` it leaves behind is deleted
automatically by the next save (e.g. when you resume).

### Milestones

`--milestones 1000,10000,100000,1000000,5000000` saves a permanent copy of the weights at
exactly those game counts, in `checkpoints/<run>/milestones/games-<count>/`. Unlike the
regular checkpoint these are never deleted. Each milestone is a complete checkpoint, so it
can be evaluated, watched (`game.play --bot`) or resumed from like any other.

- Per run: 5 × 256 MB ≈ 1.3 GB, so ≈ 3.8 GB for all three runs (plus 3 × 256 MB for the
  latest checkpoints).
- Expected timing with 3 runs in parallel: 1k/10k/100k within the first ~2 minutes, 1 M after
  ~0.5 h, 5 M after ~3.5–4 h. 5 M is reached unless the runs are much slower than measured.
- Milestones are part of the run's settings: a resumed run keeps saving the ones it has not
  reached yet. If a run is resumed from a checkpoint older than a milestone it already
  saved, that milestone is rewritten with identical weights.

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

- Evaluate or watch a milestone (pass its directory):

  ```bash
  ls checkpoints/overnight-a01/milestones/
  ~/.local/bin/uv run python -m ntuple.evaluate --checkpoint checkpoints/overnight-a01/milestones/games-0001000000 --games 1000 --depth 0
  ~/.local/bin/uv run python -m game.play --bot checkpoints/overnight-a01/milestones/games-0001000000 --depth 2
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
