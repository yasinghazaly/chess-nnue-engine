"""Turn lichess_db_eval.jsonl into arrays the NNUE trainer can read.

    python parse_evals.py eval_20M.jsonl train_data

Needs numpy and nothing else. No numba, no python-chess, no torch.

Writes three files:

    train_data_feat.npy   int16 (N, 32)  active feature indices, -1 padded
    train_data_stm.npy    int8  (N,)     side to move: 0 white, 1 black
    train_data_cp.npy     int16 (N,)     Stockfish score, from WHITE's side

Feature index, identical to the kernel that will run inside the agent:

    index = colour * 384 + (piece_type - 1) * 64 + square
    colour 0 = white; piece_type 1..6 = P N B R Q K; square 0 = a1, 63 = h8

Rows are dropped when:
  * the best line is a forced mate -- a mate is not a centipawn score, and
    the search already handles mates perfectly well
  * |cp| exceeds CP_CLIP -- the game is decided, so the number is noise
  * the side to move is in check -- the engine never asks a static evaluator
    to judge one of these, so training on them is off-task
  * the deepest available evaluation is shallower than MIN_DEPTH
  * the row is malformed or does not have exactly two kings

Memory is bounded: rows go into fixed-size numpy blocks that are concatenated
at the end, so twenty million rows cost about 1.3 GB rather than the 8 GB a
Python list of lists would need.
"""

import json
import sys
import time

import numpy as np

CP_CLIP = 3000
MIN_DEPTH = 8
MAX_FEATURES = 32
BLOCK = 1_000_000
REPORT_EVERY = 1_000_000

PIECE = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}

# ---------------------------------------------------------------------------
# Attack tables, built once at import, used only by the in-check test.
# ---------------------------------------------------------------------------
KNIGHT_ATT = [0] * 64
KING_ATT = [0] * 64
PAWN_ATT = [[0] * 64, [0] * 64]
for _s in range(64):
    _r, _f = divmod(_s, 8)
    for _dr, _df in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                     (1, 2), (1, -2), (-1, 2), (-1, -2)):
        _rr, _ff = _r + _dr, _f + _df
        if 0 <= _rr < 8 and 0 <= _ff < 8:
            KNIGHT_ATT[_s] |= 1 << (_rr * 8 + _ff)
    for _dr in (-1, 0, 1):
        for _df in (-1, 0, 1):
            if _dr or _df:
                _rr, _ff = _r + _dr, _f + _df
                if 0 <= _rr < 8 and 0 <= _ff < 8:
                    KING_ATT[_s] |= 1 << (_rr * 8 + _ff)
    for _df in (-1, 1):
        _ff = _f + _df
        if 0 <= _ff < 8:
            if _r + 1 < 8:
                PAWN_ATT[0][_s] |= 1 << ((_r + 1) * 8 + _ff)
            if _r - 1 >= 0:
                PAWN_ATT[1][_s] |= 1 << ((_r - 1) * 8 + _ff)

BISHOP_RAYS = [[] for _ in range(64)]
ROOK_RAYS = [[] for _ in range(64)]
for _s in range(64):
    _r, _f = divmod(_s, 8)
    for _dr, _df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _ray = []
        _rr, _ff = _r + _dr, _f + _df
        while 0 <= _rr < 8 and 0 <= _ff < 8:
            _ray.append(_rr * 8 + _ff)
            _rr += _dr
            _ff += _df
        if _ray:
            BISHOP_RAYS[_s].append(_ray)
    for _dr, _df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _ray = []
        _rr, _ff = _r + _dr, _f + _df
        while 0 <= _rr < 8 and 0 <= _ff < 8:
            _ray.append(_rr * 8 + _ff)
            _rr += _dr
            _ff += _df
        if _ray:
            ROOK_RAYS[_s].append(_ray)


def _hits(mask, board, enemy, wanted):
    while mask:
        low = mask & -mask
        sq = low.bit_length() - 1
        mask ^= low
        piece = board.get(sq)
        if piece is not None and piece[0] == enemy and piece[1] in wanted:
            return True
    return False


def in_check(board, king_sq, side):
    """True when `side`'s king on `king_sq` is attacked."""
    enemy = 1 - side
    if _hits(KNIGHT_ATT[king_sq], board, enemy, (2,)):
        return True
    # A square our own king would attack as a pawn is exactly a square an
    # enemy pawn could be attacking us from.
    if _hits(PAWN_ATT[side][king_sq], board, enemy, (1,)):
        return True
    if _hits(KING_ATT[king_sq], board, enemy, (6,)):
        return True
    for ray in BISHOP_RAYS[king_sq]:
        for sq in ray:
            piece = board.get(sq)
            if piece is not None:
                if piece[0] == enemy and (piece[1] == 3 or piece[1] == 5):
                    return True
                break
    for ray in ROOK_RAYS[king_sq]:
        for sq in ray:
            piece = board.get(sq)
            if piece is not None:
                if piece[0] == enemy and (piece[1] == 4 or piece[1] == 5):
                    return True
                break
    return False


def parse_fen(fen):
    """Return (features, board, king square of side to move, side) or None."""
    space = fen.find(" ")
    if space < 0 or len(fen) <= space + 1:
        return None
    side = 0 if fen[space + 1] == "w" else 1
    feats = []
    board = {}
    kings = [None, None]
    rank, file_ = 7, 0
    for ch in fen[:space]:
        if ch == "/":
            rank -= 1
            file_ = 0
        elif "1" <= ch <= "8":
            file_ += ord(ch) - 48
        else:
            lower = ch.lower()
            pt = PIECE.get(lower)
            if pt is None or rank < 0 or file_ > 7:
                return None
            colour = 0 if ch != lower else 1
            sq = rank * 8 + file_
            feats.append(colour * 384 + (pt - 1) * 64 + sq)
            board[sq] = (colour, pt)
            if pt == 6:
                kings[colour] = sq
            file_ += 1
    if kings[0] is None or kings[1] is None or len(feats) > MAX_FEATURES:
        return None
    return feats, board, kings[side], side


def best_cp(record):
    """Score of the deepest available evaluation's first principal variation."""
    evals = record.get("evals")
    if not evals:
        return None
    depth_best = -1
    cp_best = None
    for entry in evals:
        pvs = entry.get("pvs")
        if not pvs:
            continue
        cp = pvs[0].get("cp")
        if cp is None:
            continue
        depth = entry.get("depth", MIN_DEPTH)
        if depth > depth_best:
            depth_best = depth
            cp_best = cp
    if cp_best is None or depth_best < MIN_DEPTH:
        return None
    return cp_best


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python parse_evals.py <input.jsonl> <output_prefix> [max_rows]")
    in_path, prefix = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    feat_blocks, stm_blocks, cp_blocks = [], [], []
    fbuf = np.full((BLOCK, MAX_FEATURES), -1, dtype=np.int16)
    sbuf = np.zeros(BLOCK, dtype=np.int8)
    cbuf = np.zeros(BLOCK, dtype=np.int16)
    fill = 0
    seen = kept = 0
    d_mate = d_big = d_check = d_bad = 0
    started = time.time()

    def flush():
        nonlocal fill
        if fill:
            feat_blocks.append(fbuf[:fill].copy())
            stm_blocks.append(sbuf[:fill].copy())
            cp_blocks.append(cbuf[:fill].copy())
            fill = 0

    with open(in_path, encoding="utf-8") as handle:
        for line in handle:
            seen += 1
            if seen % REPORT_EVERY == 0:
                rate = seen / max(time.time() - started, 1e-9)
                print(f"  {seen:>12,} read | {kept:>12,} kept "
                      f"({100 * kept / seen:4.1f}%) | {rate / 1000:5.0f}k/s",
                      flush=True)
            if limit and kept >= limit:
                break
            try:
                record = json.loads(line)
            except ValueError:
                d_bad += 1
                continue
            cp = best_cp(record)
            if cp is None:
                d_mate += 1
                continue
            if cp > CP_CLIP or cp < -CP_CLIP:
                d_big += 1
                continue
            parsed = parse_fen(record.get("fen", ""))
            if parsed is None:
                d_bad += 1
                continue
            features, board, king_sq, side = parsed
            if in_check(board, king_sq, side):
                d_check += 1
                continue
            n = len(features)
            fbuf[fill, :n] = features
            fbuf[fill, n:] = -1
            sbuf[fill] = side
            cbuf[fill] = cp
            fill += 1
            kept += 1
            if fill == BLOCK:
                flush()

    flush()
    feat = (np.concatenate(feat_blocks) if feat_blocks
            else np.zeros((0, MAX_FEATURES), dtype=np.int16))
    stm = np.concatenate(stm_blocks) if stm_blocks else np.zeros(0, dtype=np.int8)
    cps = np.concatenate(cp_blocks) if cp_blocks else np.zeros(0, dtype=np.int16)

    np.save(prefix + "_feat.npy", feat)
    np.save(prefix + "_stm.npy", stm)
    np.save(prefix + "_cp.npy", cps)

    mins = (time.time() - started) / 60
    print(f"\nread {seen:,}, kept {kept:,} "
          f"({100 * kept / max(seen, 1):.1f}%) in {mins:.1f} min")
    print(f"  dropped: mate {d_mate:,} | |cp|>{CP_CLIP} {d_big:,} | "
          f"in check {d_check:,} | malformed {d_bad:,}")
    print(f"  wrote {feat.nbytes / 1e6:.0f} MB of features")

    if kept == 0:
        sys.exit("\nnothing kept -- do not train")

    counts = (feat >= 0).sum(1)
    print("\nsanity")
    print(f"  pieces per position : min {counts.min()}, max {counts.max()}, "
          f"mean {counts.mean():.1f}   (expect 2 .. 32)")
    print(f"  feature index range : {feat[feat >= 0].min()} .. "
          f"{feat[feat >= 0].max()}   (must be 0 .. 767)")
    print(f"  score               : mean {cps.mean():+.0f}, "
          f"median {np.median(cps):+.0f}, sd {cps.std():.0f}")
    print(f"  white to move       : {100 * (stm == 0).mean():.1f}%")

    white_q = np.isin(feat, np.arange(4 * 64, 5 * 64)).sum(1)
    black_q = np.isin(feat, np.arange(384 + 4 * 64, 384 + 5 * 64)).sum(1)
    up, down = white_q > black_q, black_q > white_q
    if up.any() and down.any():
        print(f"\n  mean cp, White up a queen : {cps[up].mean():+6.0f}")
        print(f"  mean cp, Black up a queen : {cps[down].mean():+6.0f}")
        if cps[up].mean() > cps[down].mean():
            print("  SIGN OK -- cp is from White's point of view")
        else:
            print("  SIGN WRONG -- stop and check this output ")


if __name__ == "__main__":
    main()
