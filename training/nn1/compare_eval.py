"""Go/no-go measurement: is the network actually better than v7's evaluation?

    python compare_eval.py NNUE

Both are scored against the same Stockfish labels, on the same held-out
positions, in centipawns from White's point of view. Three numbers come out:

    material only   -- the floor. Counting wood and nothing else.
    numbav7 HCE     -- what the engine uses today.
    NNUE (int16)    -- the shipped quantised network, not the float one.

If the network does not clearly beat numbav7 here, it will not beat it over
the board either, and the sensible move is to stop and ship v7.

Board state is rebuilt from the parser's feature indices, so this uses
exactly the positions the network trained against and needs no re-parse:

    index = colour * 384 + (piece_type - 1) * 64 + square

Needs numpy, numba and numbav7.py in the same folder.
"""

import sys
import time

import numpy as np

import numbav7 as V7

SAMPLE = 200_000
PAD_INDEX = 769
STM_INDEX = 768
QA, QB = 127, 64

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int64)


def rebuild(feat_row):
    """Feature indices -> (bb, mb), the representation numbav7 evaluates."""
    bb = np.zeros(8, dtype=np.uint64)
    mb = np.zeros(64, dtype=np.int8)
    for f in feat_row:
        if f < 0 or f >= 768:
            continue
        colour = f // 384
        rest = f % 384
        piece = rest // 64 + 1
        square = rest % 64
        one = np.uint64(1) << np.uint64(square)
        bb[colour] |= one
        bb[piece + 1] |= one
        mb[square] = piece
    return bb, mb


def material(feat_row):
    total = 0
    for f in feat_row:
        if f < 0 or f >= 768:
            continue
        colour = f // 384
        piece = (f % 384) // 64 + 1
        total += PIECE_VALUE[piece] if colour == 0 else -PIECE_VALUE[piece]
    return total


def nnue_eval(feat_row, stm, w1, b1, w2, b2):
    acc = b1.astype(np.int32).copy()
    for f in feat_row:
        if f >= 0:
            acc += w1[f]
    if stm == 1:
        acc += w1[STM_INDEX]
    acc = np.clip(acc, 0, QA)
    return (int((acc * w2.astype(np.int32)).sum()) + int(b2)) // (QA * QB)


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "NNUE"

    feat = np.load(prefix + "_feat.npy", mmap_mode="r")
    stm = np.load(prefix + "_stm.npy", mmap_mode="r")
    cps = np.load(prefix + "_cp.npy", mmap_mode="r")

    # The trainer's validation split, reproduced exactly: same seed, same
    # permutation, same first 1%. So these are positions the network never
    # trained on, which is the only fair comparison.
    import torch
    n = feat.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0)).numpy()
    val_idx = np.sort(order[:max(int(n * 0.01), 1)])[:SAMPLE]
    print(f"{len(val_idx):,} held-out positions "
          f"(the trainer's validation split)\n")

    rows = np.asarray(feat[val_idx], dtype=np.int64)
    sides = np.asarray(stm[val_idx], dtype=np.int64)
    target = np.asarray(cps[val_idx], dtype=np.float64)

    w1 = np.load("nnue_w1.npy")
    b1 = np.load("nnue_b1.npy")
    w2 = np.load("nnue_w2.npy")
    b2 = np.load("nnue_b2.npy")

    print("scoring... (numbav7 compiles first, about a minute)")
    started = time.time()

    mat = np.empty(len(rows))
    hce = np.empty(len(rows))
    net = np.empty(len(rows))
    for i, row in enumerate(rows):
        bb, mb = rebuild(row)
        mat[i] = material(row)
        # evaluate_raw returns from `side`'s point of view and adds TEMPO;
        # ask for White's view and take the tempo back off.
        hce[i] = V7.evaluate_raw(bb, mb, 0) - V7.TEMPO
        net[i] = nnue_eval(row, sides[i], w1, b1, w2, b2)
        if i and i % 25_000 == 0:
            print(f"  {i:,} / {len(rows):,}", flush=True)

    print(f"done in {(time.time()-started)/60:.1f} min\n")

    def report(name, pred):
        err = np.abs(pred - target)
        corr = np.corrcoef(pred, target)[0, 1]
        print(f"  {name:<16} MAE {err.mean():6.1f} cp   "
              f"median {np.median(err):6.1f}   "
              f"correlation {corr:.3f}")

    print(f"target spread: sd {target.std():.0f} cp\n")
    report("material only", mat)
    report("numbav7 HCE", hce)
    report("NNUE int16", net)

    print()
    if net.mean() == 0 or np.isnan(net).any():
        print("  network output looks broken -- stop and check this output")
    elif np.abs(net - target).mean() < np.abs(hce - target).mean() * 0.9:
        print("  VERDICT: network clearly ahead. Worth building v8.")
    elif np.abs(net - target).mean() < np.abs(hce - target).mean():
        print("  VERDICT: network ahead, but not by much. Marginal.")
    else:
        print("  VERDICT: v7's evaluation is BETTER. Stop and ship v7.")


if __name__ == "__main__":
    main()