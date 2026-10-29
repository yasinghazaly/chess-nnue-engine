"""Gate v2 -- decides whether a trained network is allowed into the engine.
 
    python gate_v2.py                    # checks nnue_v2.bin against bench_v2.npz
    python gate_v2.py some_other.bin
 
Numpy only. Scores the network exactly as numbav9 would run it: int16
quantised weights, integer arithmetic. Every criterion comes from a failure
we have actually seen:
 
  1. Correlation with Stockfish on quiet positions from our own games, where
     the first network lost to the hand-written evaluation (and to counting
     material on the strictly quiet set).
  2. Game 90, move 12: a knight down for a pawn. v8 said -54, Stockfish -284.
  3. Game 90, move 30: Black up bishop and knight for a pawn, pieces on the
     back rank. v8 said +238 for White, Stockfish -439.
  4. Queen against bare king must look won (v8: +444).
  5. Exact colour symmetry, and quantisation accurate to a few centipawns.
 
The side-to-move bonus is reported, not gated: Stockfish's own bonus on these
positions has a median of +69 cp, so there is no small "correct" number.
"""
from __future__ import annotations
 
import json
import struct
import sys
 
import numpy as np
 
QA, QB = 510, 4096      # accumulator scale, output-weight scale (int32 acc, int64 output sum)
MAGIC = b"NNUEV2\x00\x00"
 
G90_M12 = "r1bqk2r/pp1nbppp/4p3/n7/2P5/2PP1NP1/P1Q2PBP/R1B2RK1 w kq - 1 12"
G90_M30 = "r1bn1bk1/pp3p1p/6p1/2P5/q2P4/2P3PP/1Q3PB1/4R1K1 w - - 2 30"
KQK = ["8/8/8/3k4/8/8/8/Q3K3 w - - 0 1", "8/8/8/3k4/8/8/8/Q3K3 b - - 0 1",
       "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "7k/8/8/8/8/8/8/K6Q b - - 0 1",
       "8/8/2k5/8/8/5Q2/8/6K1 w - - 0 1", "8/8/2k5/8/8/5Q2/8/6K1 b - - 0 1"]
INFO = [("KR vs K", "8/8/8/3k4/8/8/8/R3K3 w - - 0 1", "+206"),
        ("KBN vs K", "8/8/8/3k4/8/8/8/1NB1K3 w - - 0 1", "+206"),
        ("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "+24"),
        ("White up a knight", "r1bqkb1r/pppp1ppp/5n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1", "+381")]
 
_PIECE = {c: ("PNBRQKpnbrqk".index(c)) for c in "PNBRQKpnbrqk"}   # code = colour*6 + type
VAL = [1, 3, 3, 5, 9, 0]
 
 
# --------------------------------------------------------------------------
def load_net(path):
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path} is not a v2 network file")
    ver, H, NB, NH, N_IN, scale, mlen = struct.unpack_from("<7i", raw, 8)
    off = 8 + 28
    bucket = list(struct.unpack_from("<64i", raw, off))
    off += 256
    info = json.loads(raw[off:off + mlen])
    off += mlen
 
    def take(count):
        nonlocal off
        a = np.frombuffer(raw, dtype="<f4", count=count, offset=off)
        off += 4 * count
        return a
 
    w1 = take(N_IN * H).reshape(N_IN, H)
    b1 = take(H)
    w2 = take(NH * 2 * H).reshape(NH, 2 * H)
    b2 = take(NH)
    assert off == len(raw), "trailing bytes in weights file"
    return dict(H=H, NB=NB, NH=NH, scale=scale, bucket=bucket, info=info, w1=w1, b1=b1, w2=w2, b2=b2,
                w1q=np.round(w1 * QA).astype(np.int32), b1q=np.round(b1 * QA).astype(np.int32),
                w2q=np.round(w2 * QB).astype(np.int64), b2q=np.round(b2 * QA * QB).astype(np.int64))
 
 
def parse_fen(fen):
    parts = fen.split()
    board = [-1] * 64
    rank, file_ = 7, 0
    for ch in parts[0]:
        if ch == "/":
            rank -= 1
            file_ = 0
        elif ch.isdigit():
            file_ += int(ch)
        else:
            board[rank * 8 + file_] = _PIECE[ch]
            file_ += 1
    return board, (0 if parts[1] == "w" else 1)
 
 
def mirror(board, stm):
    """Flip vertically and swap colours: the same position from the other side."""
    out = [-1] * 64
    for sq, p in enumerate(board):
        if p >= 0:
            out[sq ^ 56] = (p + 6) % 12
    return out, 1 - stm
 
 
def attacked_king(board, side):
    """Is `side`'s king attacked by the other side?"""
    ks = board.index(side * 6 + 5)
    o = (1 - side) * 6
    r, f = divmod(ks, 8)
    pawn_from = [(1, -1), (1, 1)] if side == 0 else [(-1, -1), (-1, 1)]
    for dr, df in pawn_from:
        rr, ff = r + dr, f + df
        if 0 <= rr < 8 and 0 <= ff < 8 and board[rr * 8 + ff] == o:
            return True
    for dr, df in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
        rr, ff = r + dr, f + df
        if 0 <= rr < 8 and 0 <= ff < 8 and board[rr * 8 + ff] == o + 1:
            return True
    for dirs, hit in (([(1, 0), (-1, 0), (0, 1), (0, -1)], (o + 3, o + 4)),
                      ([(1, 1), (1, -1), (-1, 1), (-1, -1)], (o + 2, o + 4))):
        for dr, df in dirs:
            rr, ff = r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                p = board[rr * 8 + ff]
                if p != -1:
                    if p in hit:
                        return True
                    break
                rr, ff = rr + dr, ff + df
    return False
 
 
def views(net, board):
    wk, bk = board.index(5), board.index(11)
    wfl = 7 if wk % 8 >= 4 else 0
    bfl = (7 if bk % 8 >= 4 else 0) ^ 56
    wb, bb = net["bucket"][wk ^ wfl], net["bucket"][bk ^ bfl]
    iw, ib = [], []
    for sq, p in enumerate(board):
        if p < 0:
            continue
        col, pt = divmod(p, 6)
        iw.append(wb * 768 + col * 384 + pt * 64 + (sq ^ wfl))
        ib.append(bb * 768 + (1 - col) * 384 + pt * 64 + (sq ^ bfl))
    head = min(net["NH"] - 1, (len(iw) - 2) // 4)
    return iw, ib, head
 
 
def eval_float(net, board, stm):
    """Centipawns, side to move's point of view."""
    iw, ib, head = views(net, board)
    aw = net["b1"] + net["w1"][iw].sum(0)
    ab = net["b1"] + net["w1"][ib].sum(0)
    us, them = (aw, ab) if stm == 0 else (ab, aw)
    x = np.clip(np.concatenate([us, them]), 0, 1) ** 2
    return float(x @ net["w2"][head] + net["b2"][head]) * net["scale"]
 
 
def eval_quant(net, board, stm):
    """Integer evaluation exactly as numbav9 will compute it (centipawns, stm POV)."""
    iw, ib, head = views(net, board)
    aw = net["b1q"] + net["w1q"][iw].sum(0)
    ab = net["b1q"] + net["w1q"][ib].sum(0)
    us, them = (aw, ab) if stm == 0 else (ab, aw)
    v = np.clip(np.concatenate([us, them]), 0, QA).astype(np.int64)
    s = int((v * v * net["w2q"][head]).sum())
    y = s // QA + int(net["b2q"][head])
    return (y * net["scale"]) // (QA * QB)
 
 
def white_pov(net, fen, quant=True):
    board, stm = parse_fen(fen)
    e = (eval_quant if quant else eval_float)(net, board, stm)
    return e if stm == 0 else -e
 
 
def material(board):
    return sum((VAL[p % 6] if p < 6 else -VAL[p % 6]) for p in board if p >= 0)
 
 
# --------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "nnue_v2.bin"
    net = load_net(path)
    d = np.load("bench_v2.npz")
    fens = [str(f) for f in d["fen"]]
    sf = d["sf_cp"].astype(float)
    k = np.abs(sf) < 1500
    quiet, strict = d["quiet"] & k, d["strict"] & k
    print(f"network  : {path}   H={net['H']} buckets={net['NB']} heads={net['NH']}   trainer info {net['info']}")
 
    # -- quantisation safety --------------------------------------------------
    worst_acc = int(np.abs(net["b1q"]).max() + 32 * np.abs(net["w1q"]).max())
    w1_max = int(np.abs(net["w1q"]).max())
    w2_max = int(np.abs(net["w2q"]).max())
    int_ok = w1_max < 32767 and w2_max < 32767 and worst_acc < 2 ** 31 - 1
 
    # -- benchmark ------------------------------------------------------------
    boards = [parse_fen(f) for f in fens]
    eq = np.array([eval_quant(net, b, s) * (1 if s == 0 else -1) for b, s in boards], float)
    ef = np.array([eval_float(net, b, s) * (1 if s == 0 else -1) for b, s in boards], float)
    qerr = float(np.abs(eq - ef).max())
    mat = np.array([material(b) for b, _ in boards], float)
 
    def corr(x, m):
        return float(np.corrcoef(x[m], sf[m])[0, 1])
 
    def mae_scaled(x, m):
        kk = np.dot(x[m], sf[m]) / np.dot(x[m], x[m])
        return float(np.mean(np.abs(kk * x[m] - sf[m])))
 
    rows = [("material x100", mat * 100), ("v7 hand-written", d["v7_hce"].astype(float)),
            ("v8 network", d["v8_nnue"].astype(float)), ("NEW network", eq)]
    print("\nbenchmark: your own games, Stockfish labels, |SF| < 1500")
    print(f"  {'':18s} {'quiet corr':>10s} {'strict corr':>11s} {'quiet MAE*':>10s} {'strict MAE*':>11s}")
    for name, x in rows:
        print(f"  {name:18s} {corr(x, quiet):10.3f} {corr(x, strict):11.3f} {mae_scaled(x, quiet):10.1f} "
              f"{mae_scaled(x, strict):11.1f}")
    print(f"  (* MAE after fitting one scale factor, so evaluations on different scales compare fairly)")
    print(f"  quiet n={quiet.sum()}, strict n={strict.sum()}")
 
    # -- symmetry and side-to-move bonus --------------------------------------
    sym_bad = 0
    tempo = []
    for b, s in boards:
        mb, ms = mirror(b, s)
        if eval_quant(net, b, s) != eval_quant(net, mb, ms):
            sym_bad += 1
        # same placement, other side to move; skip if that side would be "in check" illegally
        if not attacked_king(b, s) and not attacked_king(b, 1 - s):
            tempo.append(eval_quant(net, b, s) + eval_quant(net, b, 1 - s))
    tempo = np.array(tempo, float)
 
    # -- probes -----------------------------------------------------------------
    m12 = white_pov(net, G90_M12)
    m30 = white_pov(net, G90_M30)
    kqk = [white_pov(net, f) for f in KQK]
 
    # -- scale and implied piece values ----------------------------------------
    slope = float(np.polyfit(mat[quiet], eq[quiet], 1)[0])
    sf_slope = float(np.polyfit(mat[quiet], sf[quiet], 1)[0])
    hce_slope = float(np.polyfit(mat[quiet], d["v7_hce"][quiet].astype(float), 1)[0])
    pv = {i: [] for i in range(5)}
    for j in np.flatnonzero(quiet)[::3]:
        b, s = boards[j]
        base = eval_quant(net, b, s) * (1 if s == 0 else -1)
        for sq, p in enumerate(b):
            if p < 0 or p % 6 == 5:
                continue
            b2 = list(b)
            b2[sq] = -1
            e2 = eval_quant(net, b2, s) * (1 if s == 0 else -1)
            pv[p % 6].append((base - e2) * (1 if p < 6 else -1))
 
    print("\nother measurements")
    print(f"  side-to-move bonus : median {np.median(tempo):+.0f} cp, mean {tempo.mean():+.0f}, "
          f"95th pct {np.percentile(tempo, 95):+.0f}   (Stockfish: median +69, mean +137; v8: +84 / +95)")
    print(f"  cp per pawn of material (quiet): network {slope:.1f}, Stockfish {sf_slope:.1f}, v7 {hce_slope:.1f}")
    print(f"  -> numbav9 output scale to keep v7's pruning margins meaningful: x{hce_slope / slope:.3f}")
    print(f"  implied piece values P/N/B/R/Q : " + " / ".join(f"{np.mean(pv[i]):.0f}" for i in range(5))
          + "   (v8: 102 / 228 / 289 / 351 / 493, v7: 100 / 320 / 330 / 500 / 900)")
    for name, fen, old in INFO:
        print(f"  {name:18s}: {white_pov(net, fen):+6d}   (v8: {old})")
    print(f"  quantisation       : max |int - float| on bench {qerr:.1f} cp; worst-case accumulator "
          f"{worst_acc} ({'fits int16' if worst_acc < 32767 else 'needs int32'}); max |w1| {w1_max}, max |w2| {w2_max}")
 
    # -- verdict ----------------------------------------------------------------
    sq_, ss_ = corr(eq, quiet), corr(eq, strict)
    checks = [
        (f"strict-quiet correlation >= 0.915 (v7 0.905)", ss_ >= 0.915, f"{ss_:.3f}"),
        (f"quiet correlation >= 0.880 (v7 0.868)", sq_ >= 0.880, f"{sq_:.3f}"),
        ("Game 90 move 12, knight for a pawn down: <= -150", m12 <= -150, f"{m12:+d}  (v8 -54, SF -284)"),
        ("Game 90 move 30, B+N for a pawn down:   <= -150", m30 <= -150, f"{m30:+d}  (v8 +238, SF -439)"),
        ("queen vs bare king, 6 positions: all >= +700", min(kqk) >= 700, f"min {min(kqk):+d}  (v8 +444)"),
        ("exact colour symmetry", sym_bad == 0, f"{sym_bad} violations"),
        ("quantised weights in range, error <= 10 cp", int_ok and qerr <= 10, f"{qerr:.1f} cp"),
    ]
    print("\nGATE")
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {val}")
    passed = all(ok for _, ok, _ in checks)
    print("\n  VERDICT: " + ("PASS"
                            if passed else "FAIL"))
    print("DONE")
 
 
if __name__ == "__main__":
    main()
