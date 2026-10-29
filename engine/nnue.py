"""Learned evaluation: weight loading, integer inference for both networks,
and the evaluate() entry point the search calls.

Weights are int16 binary files in weights/, checked at import by CRC, size
and overflow headroom. They must be loaded in this module, before any
kernel compiles, because Numba reads a global array when it compiles the
function that uses it. If a load fails the flag stays 0 and evaluation
falls back: third network -> first network -> hand-written evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from .config import *  # noqa: F401,F403
from .board import *  # noqa: F401,F403
from .hce import *  # noqa: F401,F403

_WEIGHTS = Path(__file__).resolve().parent / "weights"

NNUE_H = 512
NNUE_QA = 127
NNUE_QB = 64
NNUE_STM = 768
NNUE_B2 = 232
NNUE_CRC = 1588365837
NNUE_OK = np.zeros(1, dtype=np.int64)
NN2_OK = np.zeros(1, dtype=np.int64)     # v10 (third) network loaded
NN2_W1G = np.zeros((NN2_IN, NN2_H), dtype=np.int16)
NN2_B1G = np.zeros(NN2_H, dtype=np.int16)
NN2_W2G = np.zeros((NN2_NH, 2 * NN2_H), dtype=np.int16)
NN2_B2G = np.zeros(NN2_NH, dtype=np.int64)
NNUE_W1 = np.zeros((770, NNUE_H), dtype=np.int16)
NNUE_B1 = np.zeros(NNUE_H, dtype=np.int16)
NNUE_W2 = np.zeros(NNUE_H, dtype=np.int16)


def _load_nnue() -> None:
    """Decode the weights and check them before letting the engine use them."""
    global NNUE_W1, NNUE_B1, NNUE_W2
    import zlib

    raw = (_WEIGHTS / "nnue_v1.bin").read_bytes()
    if zlib.crc32(raw) != NNUE_CRC:
        raise ValueError("NNUE weight checksum mismatch")

    n1 = 770 * NNUE_H * 2
    n2 = NNUE_H * 2
    if len(raw) != n1 + n2 + n2:
        raise ValueError(f"NNUE weight blob is {len(raw)} bytes, expected {n1 + 2 * n2}")

    w1 = np.frombuffer(raw[:n1], dtype=np.int16).reshape(770, NNUE_H).copy()
    b1 = np.frombuffer(raw[n1:n1 + n2], dtype=np.int16).copy()
    w2 = np.frombuffer(raw[n1 + n2:], dtype=np.int16).copy()

    # The padding row must be zero or a position with fewer than 32 pieces
    # would have garbage added to its accumulator.
    if w1[769].any():
        raise ValueError("NNUE padding row is not zero")
    # int16 accumulator headroom, checked rather than assumed.
    worst = int(np.abs(b1).max()) + 33 * int(np.abs(w1).max())
    if worst >= 32767:
        raise ValueError(f"NNUE accumulator can overflow int16: {worst}")

    NNUE_W1, NNUE_B1, NNUE_W2 = w1, b1, w2
    NNUE_OK[0] = 1


try:
    _load_nnue()
except Exception as _nnue_error:  # pragma: no cover - defensive only
    print(f"NNUE LOAD FAILED, falling back to the v7 evaluation: {_nnue_error!r}")


# v10: the third network (nnue_v3.bin, written by train_v3.py). int16 W1 | int16 B1 | int16 W2 | int32 B2, little-
# endian, zlib then base64. Loaded before warm-up: numba reads global arrays
# when it compiles, so the weights and NN2_OK must be in place by then.
NN2_CRC = 756623691


def _load_nnue2() -> None:
    import zlib

    raw = (_WEIGHTS / "nnue_v3.bin").read_bytes()
    if zlib.crc32(raw) != NN2_CRC:
        raise ValueError("NNUE v3 weight checksum mismatch")
    n1, n2, n3, n4 = NN2_IN * NN2_H, NN2_H, NN2_NH * 2 * NN2_H, NN2_NH
    expect = 2 * (n1 + n2 + n3) + 4 * n4
    if len(raw) != expect:
        raise ValueError(f"NNUE v3 blob is {len(raw)} bytes, expected {expect}")
    o = 0
    w1 = np.frombuffer(raw, dtype="<i2", count=n1, offset=o)
    o += 2 * n1
    b1 = np.frombuffer(raw, dtype="<i2", count=n2, offset=o)
    o += 2 * n2
    w2 = np.frombuffer(raw, dtype="<i2", count=n3, offset=o)
    o += 2 * n3
    b2 = np.frombuffer(raw, dtype="<i4", count=n4, offset=o)
    # int16 accumulator and int32 product headroom, checked rather than assumed
    worst = int(np.abs(b1).max()) + 32 * int(np.abs(w1).max())
    if worst >= 32767:
        raise ValueError(f"NNUE v3 int16 accumulator could overflow: {worst}")
    if int(np.abs(w2).max()) >= 8256:
        raise ValueError("NNUE v3 output weights too large for int32 products")
    global NN2_W1G, NN2_B1G, NN2_W2G, NN2_B2G
    NN2_W1G = w1.reshape(NN2_IN, NN2_H).astype(np.int16)
    NN2_B1G = b1.astype(np.int16)
    NN2_W2G = w2.reshape(NN2_NH, 2 * NN2_H).astype(np.int16)
    NN2_B2G = b2.astype(np.int64)
    NN2_OK[0] = 1


if USE_NNUE_V2:
    try:
        _load_nnue2()
    except Exception as _nn2_error:  # pragma: no cover - defensive only
        print(f"NNUE V2 LOAD FAILED, falling back to the v8 network: {_nn2_error!r}")


@njit(cache=False)
def nnue_raw(bb, mb, side):
    """Quantised forward pass of the learned evaluation, White's point of view.

    769 inputs -> 512 hidden -> 1. Only about 32 inputs are ever set, so layer
    one is a sparse lookup and add rather than a matrix multiply. The active
    feature indices are gathered first and the arithmetic runs afterwards over
    a countable loop; written the other way round, with the adds inside the
    bitboard walk, LLVM cannot prove the trip count and emits scalar code that
    measured ten times slower.

    The accumulator is allocated here rather than passed in. That measured
    402 ns against 601 ns for a threaded buffer, because a local array cannot
    alias and numba optimises accordingly.

    int16 throughout: worst-case accumulator is 8581 against a limit of 32767.
    """
    idx = np.empty(33, dtype=np.int64)
    n = 0
    for col in range(2):
        occ = bb[col]
        while occ:
            sq = lsb(occ)
            occ &= occ - U1
            idx[n] = col * 384 + (mb[sq] - 1) * 64 + sq
            n += 1
    if side == 1:
        # Input 768 marks Black to move. Tempo is worth roughly 20 cp and the
        # network learned it here, so no TEMPO is added on top.
        idx[n] = NNUE_STM
        n += 1

    acc = np.empty(NNUE_H, dtype=np.int16)
    for i in range(NNUE_H):
        acc[i] = NNUE_B1[i]
    for k in range(n):
        row = NNUE_W1[idx[k]]
        for i in range(NNUE_H):
            acc[i] += row[i]

    total = 0
    for i in range(NNUE_H):
        v = acc[i]
        if v < 0:
            v = 0
        elif v > NNUE_QA:
            v = NNUE_QA
        total += v * NNUE_W2[i]
    return (total + NNUE_B2) // (NNUE_QA * NNUE_QB)


@njit(cache=False, inline="always")
def nn2_bucket(s):
    """Eight buckets of the (flipped, mirrored) king square: rank 1 a-b | c | d,
    rank 2 a-b | c-d, rank 3, rank 4, rank 5 and up. Same table as train_v3."""
    r = s >> 3
    f = s & 7
    if r == 0:
        if f <= 1:
            return 0
        return 1 if f == 2 else 2
    if r == 1:
        return 3 if f <= 1 else 4
    if r == 2:
        return 5
    if r == 3:
        return 6
    return 7


@njit(cache=False)
def nnue2_raw(bb, mb, side):
    """Main network, side to move's point of view, unscaled centipawns.
    In v10 this is the third network; v9 ran the second network through the
    same code with smaller constants.

    Bit-identical to gate_v2.eval_quant and to training/regate.py. Each side
    sees the board as White: Black's view is flipped vertically with colours
    swapped, and either view is mirrored left-right when that side's king is
    on files e-h.

    int16 weights and int16 accumulators: the loader proves the worst case
    (|B1| + 32 x |W1|) stays below 32767. Output products v*v*w fit int32
    (510^2 x |w2| < 2^31 for |w2| < 8256, also checked); the sum is int64.
    Measured inside compiled code for the second network (v9): 552 ns per
    evaluation against v8's 336.
    Storing the weights as int32 in the state buffer instead measured 2958 ns
    because the inner loop stopped vectorising.
    """
    wk = lsb(bb[KING + 1] & bb[0])
    bk = lsb(bb[KING + 1] & bb[1])
    wfl = 7 if (wk & 7) >= 4 else 0
    bfl = (7 if (bk & 7) >= 4 else 0) ^ 56
    wbk = nn2_bucket(wk ^ wfl)
    bbk = nn2_bucket(bk ^ bfl)
    iw = np.empty(32, dtype=np.int64)
    ib = np.empty(32, dtype=np.int64)
    n = 0
    for col in range(2):
        occ = bb[col]
        while occ:
            sq = lsb(occ)
            occ &= occ - U1
            pt = mb[sq] - 1
            iw[n] = wbk * 768 + col * 384 + pt * 64 + (sq ^ wfl)
            ib[n] = bbk * 768 + (1 - col) * 384 + pt * 64 + (sq ^ bfl)
            n += 1

    aw = NN2_B1G.copy()
    ab = NN2_B1G.copy()
    for k in range(n):
        rw = NN2_W1G[iw[k]]
        rb = NN2_W1G[ib[k]]
        for i in range(NN2_H):
            aw[i] += rw[i]
            ab[i] += rb[i]

    head = (n - 2) // 4
    if head > NN2_NH - 1:
        head = NN2_NH - 1
    if head < 0:
        head = 0
    w2 = NN2_W2G[head]
    if side == 0:
        us = aw
        them = ab
    else:
        us = ab
        them = aw
    s = np.int64(0)
    for i in range(NN2_H):
        v = np.int32(min(max(np.int32(us[i]), 0), NN2_QA))
        s += np.int64(v * v * np.int32(w2[i]))
    for i in range(NN2_H):
        v = np.int32(min(max(np.int32(them[i]), 0), NN2_QA))
        s += np.int64(v * v * np.int32(w2[NN2_H + i]))
    y = s // NN2_QA + NN2_B2G[head]
    return (y * 400) // (NN2_QA * NN2_QB)


@njit(cache=False)
def evaluate_raw(bb, mb, side):
    if USE_NNUE_V2 and NN2_OK[0] == 1:
        v = nnue2_raw(bb, mb, side) * NN2_SCALE_NUM // NN2_SCALE_DEN
        if v > NN2_CLAMP:
            v = NN2_CLAMP
        elif v < -NN2_CLAMP:
            v = -NN2_CLAMP
        return v
    if USE_NNUE and NNUE_OK[0] == 1:
        score = nnue_raw(bb, mb, side)
        if side == 1:
            score = -score
        # No opposite-bishop scaling on top of the network: it was trained
        # against Stockfish, which already knows those endings are drawish,
        # and applying the v7 factor as well would double-count. Flip
        # USE_NNUE off to get the v7 behaviour back.
        return score
    phase = (popcount(bb[KNIGHT + 1]) + popcount(bb[BISHOP + 1])
             + 2 * popcount(bb[ROOK + 1]) + 4 * popcount(bb[QUEEN + 1]))
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = colour_score(bb, mb, 0, phase) - colour_score(bb, mb, 1, phase)
    # Applied to the colour difference, before the side flip and before the
    # tempo bonus, so the scaling touches the assessment of the position and
    # nothing else.
    if USE_OCB_SCALE:
        n = ocb_scale_num(bb)
        if n != OCB_SCALE_DEN:
            score = score * n // OCB_SCALE_DEN
    if side == 1:
        score = -score
    return score + TEMPO


@njit(cache=False)
def evaluate(bb, mb, side, key, u64, i32):
    """Memoised on the Zobrist key, which is a superset of everything
    colour_score reads, so the cache is exact rather than approximate."""
    idx = np.int64(key & EV_MASK)
    if u64[EV_KEY_OFF + idx] == key:
        return np.int64(i32[EV_VAL_OFF + idx])
    v = evaluate_raw(bb, mb, side)
    u64[EV_KEY_OFF + idx] = key
    i32[EV_VAL_OFF + idx] = v
    return v
