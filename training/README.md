# Training pipelines

One folder per network. They hold different stages because each network went
through a different process.

| Folder | Network | Used in | Stages | Files |
|---|---|---|---|---|
| [`nn1/`](nn1/) | 769 → 512 → 1 | version 8; fallback in versions 9 and 10 | download, parse, train, compare with the hand-written evaluation | `download_evals.py`, `parse_evals.py`, `train.py`, `compare_eval.py` |
| [`nn2/`](nn2/) | 2 × (4 king buckets × 768 → 256), 8 heads | version 9 | parse, train, 2 × 2 sweep, gate | `parse_v2.py`, `train_v2.py`, `sweep_v2.py`, `gate_v2.py`, `bench_v2.npz`, `nnue_v2_int16.bin` |
| [`nn3/`](nn3/) | 2 × (8 king buckets × 768 → 384), 8 heads | **version 10 (this repository's engine)** | parse, train | `parse_v3.py`, `train_v3.py` |
| [`regate.py`](regate.py) | second and third | | gate checks on the int16 weights of both | |

Why they differ:

- **nn1** read a downloaded copy of the Lichess file, so downloading was a separate
  script. `compare_eval.py` was its check against the hand-written evaluation,
  written before the gate existed. It imports an older single-file engine that is
  not in this repository.
- **nn2** is where the training recipe was settled: the 2 × 2 sweep and the gate
  were built around it. Its parser streams the Lichess file directly.
  `nnue_v2_int16.bin` is its int16 weights as version 9 shipped them, kept so the gate
  tables can be reproduced. `sweep_v2.py` also needs `select_v2.npz`, which is
  not included.
- **nn3** reuses the recipe that won the sweep (run D: clamped labels dropped
  unless the winner is three pawns ahead, K = 800) on a larger network and
  120,888,098 positions. `train_v3.py` writes `nnue_v3.bin` in the engine's format,
  which the engine loads from `engine/weights/` as is.
- **`regate.py`** applies `gate_v2.py`'s checks to the int16 weights of both
  bucketed networks. `gate_v2.py` itself reads the trainers' float weights, which
  are not in this repository.

Training data (the parsed shards) is never committed.
