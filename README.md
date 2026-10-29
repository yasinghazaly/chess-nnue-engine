# chess-nnue-engine

A chess engine with a neural-network evaluation (NNUE), made for the
[AI Chessathon](https://aichessathon.com) competition and its
[starter harness](https://github.com/advitrocks9/aichessathon-starter). The engine
must run on one CPU core with 2 GB of memory and no GPU, under a clock of
120 s + 0.5 s per move.

This repository holds version 10: the engine, the evaluation network it plays
with, the scripts that trained and checked the networks, head-to-head match
results against earlier versions, and tests.

## Method

### 1. Data

Positions come from the [Lichess evaluation database](https://database.lichess.org/#evals),
where each position carries Stockfish evaluations. The file is streamed and
decompressed on the fly (zstandard) and never stored whole. For each position the
deepest evaluation is used, and:

- positions where the side to move is in check are dropped (search never asks
  for a static evaluation there);
- positions whose best move is a capture, en passant or a promotion are dropped,
  so the label describes a settled position rather than material about to change;
- mates are kept as ±2000 and scores are clamped to ±2000;
- a label at the clamp survives only if the winning side is also at least three
  pawns ahead in material (the outcome of the experiment in section 3).

The engine's network was trained on 120,888,098 positions kept this way.
Parser: [`training/nn3/parse_v3.py`](training/nn3/parse_v3.py) (the second
network's parser, [`training/nn2/parse_v2.py`](training/nn2/parse_v2.py), applies
the first three rules and a minimum search depth of 16).

### 2. The network the engine plays with

| | |
|---|---|
| Inputs | 8 king buckets × 768 piece-square features, seen from each side's point of view with shared weights; the board is mirrored so each king is on files a–d |
| Hidden layer | 384 per side, squared clipped ReLU; the side to move's half goes first (768 total) |
| Output | 8 heads, chosen by the number of pieces on the board |
| Loss | squared error in win-probability space, (σ(prediction / K) − σ(target / K))², target = Stockfish centipawns, K = 800 (section 3) |
| Training | Adam, learning rate warmed up then cosine-decayed; a shared 768-row factoriser is added to every bucket during training and merged in at export, so rarely seen king positions still learn |
| Inference | int16 weights (scales 510 and 4096), integer arithmetic only; weights clipped during training so the int16 accumulator cannot overflow |

This is a standard NNUE form, the same as the second network (version 9) with
twice the king buckets and a wider hidden layer. The weights were trained in this
project. Trainer: [`training/nn3/train_v3.py`](training/nn3/train_v3.py).
Weights: [`engine/weights/nnue_v3.bin`](engine/weights/nnue_v3.bin).

On the trainer's held-out positions, after 10 minutes of training the network
reached a loss of 0.003024 and a mean error of 112.8 cp, against 0.003264 and
115.8 cp for the second network on the same positions; the mean error was
110.9 cp at about 27 minutes of the 90-minute run (working log).

### 3. Why lower training loss gave a worse evaluation

The first long training run of the second network kept improving its held-out
loss at every checkpoint, while its error in centipawns rose from 198 to 205 and
its agreement with Stockfish on positions from the engine's own games peaked after
two minutes and then fell below the hand-written evaluation. Two causes were
suspected:

1. **Mating-attack labels.** About 34.8 M labels sat at the ±2000 clamp, 31 M of
   them mates, many with roughly equal material. A static evaluation cannot see a
   forced mate; fitting one teaches the network to call any exposed king lost.
2. **Loss saturation.** σ(x / 400) is almost flat beyond ±800 cp, so the loss
   barely distinguishes +700 from +1800.

[`training/nn2/sweep_v2.py`](training/nn2/sweep_v2.py) tests both in a 2 × 2
design, with the same seed and length for every run:

| Run | Drop clamped labels unless the winner is ≥ 3 pawns ahead | Loss scale K |
|---|---|---|
| A (control) | no | 400 |
| B | yes | 400 |
| C | no | 800 |
| D | yes | 800 |

Checkpoints were chosen on a separate selection set of positions from the
engine's self-play games, never on the gate's benchmark. Recorded result:
dropping the labels raised the selection score by 0.010 correlation; both changes
together by 0.012. The best checkpoint came from run D, which became the second
network; run D's data rule and loss are the ones the third network was trained
with.

*Sources:* the observation and the label counts are recorded in the docstring of
`sweep_v2.py`; the 2 × 2 result is from the working log at the time (the
selection set is not in this repository).

### 4. The gate

[`training/nn2/gate_v2.py`](training/nn2/gate_v2.py) sets the checks a network
must pass before it replaces the hand-written evaluation. The benchmark,
[`training/nn2/bench_v2.npz`](training/nn2/bench_v2.npz), is 3,071 positions from
the engine's own games labelled by Stockfish; 2,042 are quiet and 854 strictly
quiet (positions with |score| ≥ 1500 excluded).

[`training/regate.py`](training/regate.py) applies the gate's checks to the int16
weights of both bucketed networks: the second network
([`training/nn2/nnue_v2_int16.bin`](training/nn2/nnue_v2_int16.bin), version 9) and the third
(the engine's):

| Correlation with Stockfish | Quiet | Strictly quiet |
|---|---|---|
| Material count × 100 | 0.826 | 0.881 |
| Version 7, hand-written evaluation | 0.868 | 0.905 |
| Version 8, first network | 0.850 | 0.843 |
| Version 9, second network | 0.905 | 0.936 |
| **Version 10, third network (this engine)** | **0.903** | **0.931** |
| Gate threshold | 0.880 | 0.915 |

| Check | Threshold | Second network | Third network |
|---|---|---|---|
| Game 90, move 12 (a knight down for a pawn) | ≤ −150 | −141: **fail** | −194: pass |
| Game 90, move 30 (bishop and knight down for a pawn) | ≤ −150 | −294: pass | −386: pass |
| Queen against bare king, 6 positions | all ≥ +700 | min +1690: pass | min +1692: pass |
| Colour symmetry (each position against its mirror) | 0 differences | 0 of 3,071: pass | 0 of 3,071: pass |
| int16 accumulator, worst case | < 32,767 | 24,138: pass | 31,493: pass |
| Quantisation error, integer against float | ≤ 10 cp | not checkable here | not checkable here |

The quantisation check needs the trainer's float weights, which are not in this
repository. The third network passes every check that can be run here.

The second network failed one check, and the failure was overridden before
version 9 was built. The analysis recorded in the working log at the time: the
hand-written evaluation it was replacing scored −128 on the same position and
would also have failed, so the threshold could not separate the two; and the check
measured a static score, while what mattered was whether the engine played the
losing move, which a replay of the game showed version 9 does not.

On this benchmark the second and third networks cannot be told apart: the
differences in correlation are 0.002 (quiet) and 0.004 (strictly quiet), with
95 % bootstrap intervals that include zero. The matches below separate them.

### 5. Head-to-head matches

Versions 7 to 10 played each other in the starter harness, unchanged, at the
harness's arena clock of 10 s + 0.1 s per move. Each match used 48 start positions
(the harness's 8 and 40 standard opening lines), each played twice with colours
swapped. The stopping rule was fixed before any game: a sequential probability
ratio test on game pairs, H0 "A is no stronger than B" against H1 "A is at least
35 Elo stronger" (the 55 % ship rule), α = β = 0.05, at least 40 games, at most
400. Script: [`matches/match.py`](matches/match.py); every game, with a summary
per match, is in [`matches/results/`](matches/results/).

| Match (A vs B) | Games | A wins–draws–losses | A score, 95 % interval | Elo difference, 95 % interval | Result |
|---|---|---|---|---|---|
| v9 vs v7 | 45 | 40–3–2 | 92.2 % (85.3–99.2) | +430 (+305 to +828) | A stronger |
| v10 vs v7 | 45 | 36–6–3 | 86.7 % (78.2–95.1) | +325 (+222 to +517) | A stronger |
| v9 vs v8 | 45 | 28–14–3 | 77.8 % (68.7–86.9) | +218 (+136 to +329) | A stronger |
| v8 vs v7 | 164 | 69–37–58 | 53.4 % (46.6–60.1) | +23 (−24 to +71) | stopped by hand before a verdict |
| v10 vs v9 | 119 | 54–31–34 | 58.4 % (50.8–66.0) | +59 (+6 to +115) | v9 not stronger; v10 ahead |

Both bucketed networks beat the hand-written evaluation clearly; the first
network did not separate from it. The third network scored 58.4 % against the
second. The v10–v9 match was run as "v9 vs v10" (the table shows it from v10's
side), and v8–v7 was stopped by hand at 164 games, when the two looked too
close to separate within the game budget.

No game crashed. 102 of the 418 games ended on time; in 84 of those the side that
ran out of time was already at least two pawns down. With those games removed,
every comparison points the same way (v10 against v9: 60.7 % over 75 games).

### 6. Engine

Python compiled with [Numba](https://numba.pydata.org/), in [`engine/`](engine/):

- **Board** ([`board.py`](engine/board.py)): bitboards, legal move generation with
  pins, castling and en passant, make/unmake, Zobrist hashing.
- **Search** ([`search.py`](engine/search.py), [`agent.py`](engine/agent.py)):
  negamax with principal-variation search, transposition table (2²² entries),
  iterative deepening with aspiration windows, quiescence search, null move,
  reverse futility pruning, razoring, late-move reductions and pruning, static
  exchange evaluation, internal iterative deepening, killer and history move
  ordering, and time management that measures the clock increment.
- **Evaluation** ([`nnue.py`](engine/nnue.py), [`hce.py`](engine/hce.py)): the
  network above, scaled by 53/100 so the search's pruning margins keep their
  meaning; the first network and the hand-written evaluation remain as fallbacks.
  Weights are checked by CRC, size and overflow bound when they load.
- **Flags** ([`config.py`](engine/config.py)): changes are behind compile-time
  flags, so each can be switched off on its own.

The competition accepts a single `agent.py`.
[`scripts/build_submission.py`](scripts/build_submission.py) builds it from the
modules and embeds the weights.

### 7. Version history

| Version | Evaluation | Note |
|---|---|---|
| 7 | hand-written | baseline for the networks |
| 8 | first network (769 → 512 → 1), [`training/nn1/`](training/nn1/) | trained on 15,859,917 positions without the capture filter; scored below version 7 on the benchmark |
| 9 | second network (4 buckets, 256 hidden), [`training/nn2/`](training/nn2/) | the network of the 2 × 2 experiment and the gate |
| **10** | **third network (above), [`training/nn3/`](training/nn3/)** | **this repository**; the search is version 9's, unchanged |

Ship rules were fixed before testing: a new network version ships at 55 % or
better against the previous version over a head-to-head match; the final version
at 50 % or better with no crashes.

## How to run

Python 3.13. From the repository root:

```bash
pip install -r requirements.txt
python -m pytest tests -v                      # 85 tests, about two minutes
python training/regate.py                      # the gate tables above
python scripts/build_submission.py             # writes build/agent.py
```

The first import compiles the engine (about a minute).

**Tests** ([`tests/`](tests/)):

- move generation reproduces published perft node counts on 6 standard positions;
- all three evaluations, and fixed-depth searches with the clock disabled, match a
  reference recorded from the original single-file version 10 engine;
- the network gives the same score for every position and its colour mirror;
- the gate script's evaluator matches the engine's on the third network, and
  reproduces version 9's scores from the second network's weights;
- the parsers keep and drop the right positions, on hand-made database lines;
- the built single file matches the same reference.

CI runs them on every push ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

**Matches** need a clone of the starter harness and the engine files; see the
docstring of [`matches/match.py`](matches/match.py).

**Training** needs the Lichess file (large; the parsers stream it) and a GPU for
reasonable speed. Training data is not in this repository. Each script's docstring
gives its commands; see also [`training/README.md`](training/README.md).

## Limits

- **Matches are between versions only**, at 10 s + 0.1 s, on a laptop running six
  games at once rather than the competition's single core. There is no rating
  against outside engines. Close pairs (v8–v7) were not resolved.
- **One run per cell** in the 2 × 2 experiment, one seed; no repeats or error bars.
- **The validation set is not independent.** Held-out positions are the tail of
  the same stream the networks trained on, not a random sample.
- **No tactical test suite.** The gate checks static evaluations and a handful of
  hand-picked positions.
- **The benchmark is small** (3,071 positions) and comes from the engine's own games.
- **Training length depends on hardware.** The trainers set the number of steps
  from a measured speed and a time budget, so a rerun on other hardware will not
  repeat a run step for step.
- Several figures above come from the working log kept at the time and are marked
  as such; they have not been re-run here.

## Acknowledgements

The match harness, baselines and competition contract come from the
[AI Chessathon starter](https://github.com/advitrocks9/aichessathon-starter) (MIT).
Position labels come from the Lichess evaluation database (CC0), computed by
Stockfish. Libraries: NumPy, Numba, PyTorch, python-chess, zstandard.

No licence has been chosen for this repository.
