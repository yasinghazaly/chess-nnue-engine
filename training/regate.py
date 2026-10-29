"""Run the gate's checks on the int16 weights of the second and third networks.

    python training/regate.py                    # from the repo root
    python training/regate.py --bench other.npz

Networks
--------
    second  training/nn2/nnue_v2_int16.bin   2 x (4 buckets x 768 -> 256), 8 heads   ran in version 9
    third   engine/weights/nnue_v3.bin 2 x (8 buckets x 768 -> 384), 8 heads   this engine (version 10)

Both files hold int16 W1 | int16 B1 | int16 W2 | int32 B2, little-endian, as
written by train_v2.py / train_v3.py. Scores are computed here with numpy, with
the same integer arithmetic as the engine's kernel (engine/nnue.py, nnue2_raw)
and gate_v2.eval_quant: same features, buckets, heads, clamps and integer
division. tests/test_engine.py checks that this evaluator and the engine's
kernel agree exactly.

gate_v2.py reads the trainer's float weights, which are not in this repository,
so its quantisation-error criterion (|int - float| <= 10 cp) is reported as not
checkable. Every other criterion and threshold is gate_v2.py's.

Inputs
------
training/nn2/bench_v2.npz, 3,071 positions from the engine's own games:
    fen      (N,)  str    position
    sf_cp    (N,)  int32  Stockfish score, White's point of view
    v7_hce   (N,)  int32  version 7 hand-written evaluation, White's point of view
    v8_nnue  (N,)  int32  version 8 (first network) evaluation, White's point of view
    quiet    (N,)  bool   quiet position
    strict   (N,)  bool   strictly quiet position
    source   (N,)  str    game log the position came from
Positions with |sf_cp| >= 1500 are excluded from the correlations, as in gate_v2.py.

Units: network scores are in the gate's units (output x 400), before the
engine's output scale (61/100 for the second network in version 9, 53/100 for
the third network here). Correlations do not depend on the scale.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QA, QB, NHEAD = 510, 4096, 8

# Probe positions and thresholds, copied from gate_v2.py.
G90_M12 = "r1bqk2r/pp1nbppp/4p3/n7/2P5/2PP1NP1/P1Q2PBP/R1B2RK1 w kq - 1 12"
G90_M30 = "r1bn1bk1/pp3p1p/6p1/2P5/q2P4/2P3PP/1Q3PB1/4R1K1 w - - 2 30"
KQK = ["8/8/8/3k4/8/8/8/Q3K3 w - - 0 1", "8/8/8/3k4/8/8/8/Q3K3 b - - 0 1",
       "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "7k/8/8/8/8/8/8/K6Q b - - 0 1",
       "8/8/2k5/8/8/5Q2/8/6K1 w - - 0 1", "8/8/2k5/8/8/5Q2/8/6K1 b - - 0 1"]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def bucket4(s: int) -> int:
    """Second network: 0 back rank a-b, 1 back rank c-d, 2 second rank, 3 further up."""
    r = s >> 3
    if r == 0:
        return 0 if (s & 7) <= 1 else 1
    return 2 if r == 1 else 3


def bucket8(s: int) -> int:
    """Third network: rank 1 a-b | c | d, rank 2 a-b | c-d, rank 3, rank 4, rank 5 and up."""
    r, f = s >> 3, s & 7
    if r == 0:
        return 0 if f <= 1 else (1 if f == 2 else 2)
    if r == 1:
        return 3 if f <= 1 else 4
    return min(r + 3, 7)


class Net:
    def __init__(self, name: str, path: Path, nb: int, h: int, bucket):
        raw = path.read_bytes()
        n1, n2, n3 = nb * 768 * h, h, NHEAD * 2 * h
        if len(raw) != 2 * (n1 + n2 + n3) + 4 * NHEAD:
            raise SystemExit(f"{path}: {len(raw)} bytes, not a {nb}-bucket {h}-wide network")
        o = 0
        self.w1 = np.frombuffer(raw, "<i2", n1, o).reshape(nb * 768, h).astype(np.int64); o += 2 * n1
        self.b1 = np.frombuffer(raw, "<i2", n2, o).astype(np.int64); o += 2 * n2
        self.w2 = np.frombuffer(raw, "<i2", n3, o).reshape(NHEAD, 2 * h).astype(np.int64); o += 2 * n3
        self.b2 = np.frombuffer(raw, "<i4", NHEAD, o).astype(np.int64)
        self.name, self.path, self.h, self.bucket, self.crc = name, path, h, bucket, zlib.crc32(raw)

    def eval_stm(self, fen: str) -> int:
        """Side to move's point of view, gate units; same arithmetic as nnue2_raw."""
        board = chess.Board(fen)
        wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
        wfl = 7 if (wk & 7) >= 4 else 0
        bfl = (7 if (bk & 7) >= 4 else 0) ^ 56
        wb, bb = self.bucket(wk ^ wfl), self.bucket(bk ^ bfl)
        iw, ib = [], []
        for sq, p in board.piece_map().items():
            col, pt = (0 if p.color else 1), p.piece_type - 1
            iw.append(wb * 768 + col * 384 + pt * 64 + (sq ^ wfl))
            ib.append(bb * 768 + (1 - col) * 384 + pt * 64 + (sq ^ bfl))
        aw = self.b1 + self.w1[iw].sum(0)
        ab = self.b1 + self.w1[ib].sum(0)
        head = min(max((len(iw) - 2) // 4, 0), NHEAD - 1)
        us, them = (aw, ab) if board.turn == chess.WHITE else (ab, aw)
        w2 = self.w2[head]
        s = int((np.clip(us, 0, QA) ** 2 * w2[:self.h]).sum() + (np.clip(them, 0, QA) ** 2 * w2[self.h:]).sum())
        y = s // QA + int(self.b2[head])
        return (y * 400) // (QA * QB)

    def eval_white(self, fen: str) -> int:
        e = self.eval_stm(fen)
        return e if fen.split()[1] == "w" else -e


SECOND = lambda: Net("second network (v9)", ROOT / "training" / "nn2" / "nnue_v2_int16.bin", 4, 256, bucket4)  # noqa: E731
THIRD = lambda: Net("third network (v10)", ROOT / "engine" / "weights" / "nnue_v3.bin", 8, 384, bucket8)  # noqa: E731


def material(fen: str) -> int:
    """White minus Black material in pawns (P1 N3 B3 R5 Q9)."""
    return sum(VAL[p.piece_type] * (1 if p.color else -1) for p in chess.Board(fen).piece_map().values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(ROOT / "training" / "nn2" / "bench_v2.npz"))
    args = ap.parse_args()

    d = np.load(args.bench)
    fens = [str(f) for f in d["fen"]]
    sf = d["sf_cp"].astype(float)
    keep = np.abs(sf) < 1500
    quiet, strict = d["quiet"] & keep, d["strict"] & keep
    nets = [SECOND(), THIRD()]
    scores = {n.name: np.array([n.eval_white(f) for f in fens], float) for n in nets}

    def corr(x, m):
        return float(np.corrcoef(x[m], sf[m])[0, 1])

    bench = Path(args.bench).resolve()
    print(f"benchmark: {bench.relative_to(ROOT).as_posix() if bench.is_relative_to(ROOT) else bench}")
    print(f"  {len(fens)} positions; quiet n={int(quiet.sum())}, strict n={int(strict.sum())} (|Stockfish| < 1500)\n")
    print(f"  {'correlation with Stockfish':26s} {'quiet':>7s} {'strict':>7s}")
    rows = [("material x100", np.array([material(f) for f in fens], float) * 100),
            ("v7 hand-written", d["v7_hce"].astype(float)),
            ("v8 first network", d["v8_nnue"].astype(float))] + [(n.name, scores[n.name]) for n in nets]
    for name, x in rows:
        print(f"  {name:26s} {corr(x, quiet):7.3f} {corr(x, strict):7.3f}")

    # Can the benchmark tell the two networks apart? Paired bootstrap over positions
    # (5,000 resamples, fixed seed). Positions come from 9 games, so these intervals
    # are if anything too narrow.
    a, b = scores[nets[0].name], scores[nets[1].name]
    rng = np.random.default_rng(0)
    print("\n  second minus third network, correlation difference and 95% bootstrap interval")
    for label, m in (("quiet", quiet), ("strict", strict)):
        idx = np.flatnonzero(m)
        diffs = []
        for _ in range(5000):
            s = rng.choice(idx, len(idx))
            diffs.append(np.corrcoef(a[s], sf[s])[0, 1] - np.corrcoef(b[s], sf[s])[0, 1])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {label:7s} {corr(a, m) - corr(b, m):+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    for net in nets:
        x = scores[net.name]
        sq, ss = corr(x, quiet), corr(x, strict)
        sym_bad = sum(net.eval_stm(f) != net.eval_stm(chess.Board(f).mirror().fen()) for f in fens)
        m12, m30 = net.eval_white(G90_M12), net.eval_white(G90_M30)
        kqk = min(net.eval_white(f) for f in KQK)
        w1_max, w2_max = int(np.abs(net.w1).max()), int(np.abs(net.w2).max())
        worst_acc = int(np.abs(net.b1).max() + 32 * w1_max)
        checks = [
            ("strict-quiet correlation >= 0.915", ss >= 0.915, f"{ss:.3f}"),
            ("quiet correlation >= 0.880", sq >= 0.880, f"{sq:.3f}"),
            ("Game 90 move 12 (knight for a pawn down) <= -150", m12 <= -150, f"{m12:+d}"),
            ("Game 90 move 30 (B+N for a pawn down) <= -150", m30 <= -150, f"{m30:+d}"),
            ("queen vs bare king, 6 positions, all >= +700", kqk >= 700, f"min {kqk:+d}"),
            ("exact colour symmetry", sym_bad == 0, f"{sym_bad} violations"),
            ("int16 weights and accumulator in range", w1_max < 32767 and w2_max < 8256 and worst_acc < 32767,
             f"max|w1| {w1_max}, max|w2| {w2_max}, worst accumulator {worst_acc}"),
        ]
        rel = net.path.relative_to(ROOT).as_posix()
        print(f"\nGATE: {net.name}, {rel} (crc32 {net.crc}); thresholds from gate_v2.py")
        for name, ok, val in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:50s} {val}")
        print(f"  [ -- ] {'quantisation error <= 10 cp':50s} not checkable: float weights not in this repository")
        print(f"  {sum(ok for _, ok, _ in checks)} of {len(checks)} checkable criteria passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
