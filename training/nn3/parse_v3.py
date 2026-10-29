"""parse_v3.py -- stream the Lichess Stockfish-evaluation database into training shards.

    python parse_v3.py                                  # 120M kept positions or 80 minutes
    python parse_v3.py --target 60000000 --minutes 45   # smaller run
    python parse_v3.py --file lichess_db_eval.jsonl.zst # parse a file you already downloaded
    touch STOP                                          # stop early; everything so far is saved

Needs: numpy, zstandard.  Nothing is stored except the kept positions.

Kept positions (the run D recipe that produced NN2):
  * deepest evaluation of each position, first principal variation
  * side to move not in check
  * Stockfish's best move is quiet: not a capture, en passant or promotion
  * mates -> +-2000, scores clamped to +-2000
  * a label at the clamp (|score| >= 1900) survives only if the winning side is
    also >= 3 pawns up in material: forced mates in level positions teach a
    static evaluation to scream at every exposed king

Output: data_v3/shard_NNN.npz
  board  uint8 (N, 32)  two squares per byte, low nibble = even square (a1 = 0)
                        0 empty, 1-6 white P N B R Q K, 7-12 black P N B R Q K
  stm    uint8 (N,)     0 = white to move, 1 = black to move
  score  int16 (N,)     Stockfish score from the side to move's point of view
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request

import numpy as np

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
CLAMP = 2000
CLAMP_LEVEL = 1900
MIN_EDGE = 3
CHUNK_LINES = 20000
SHARD_SIZE = 4_000_000

CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
        "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}
VAL = (0, 1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0)          # by code
SIGN = (0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1)


def _tables():
    knight, king, rays = [], [], []
    for sq in range(64):
        r, f = divmod(sq, 8)
        knight.append([(r + dr) * 8 + f + df for dr, df in
                       ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
                       if 0 <= r + dr < 8 and 0 <= f + df < 8])
        king.append([(r + dr) * 8 + f + df for dr in (-1, 0, 1) for df in (-1, 0, 1)
                     if (dr or df) and 0 <= r + dr < 8 and 0 <= f + df < 8])
        per = []
        for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ray, rr, ff = [], r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                ray.append(rr * 8 + ff)
                rr += dr
                ff += df
            per.append(ray)
        rays.append(per)
    return knight, king, rays


KNIGHT_T, KING_T, RAYS = _tables()


def in_check(b, white):
    """Is the side given by `white` in check?  b is a 64-list of codes."""
    k = b.index(6 if white else 12)
    off = 6 if white else 0                     # enemy code offset
    en, eb, er, eq, ek = 2 + off, 3 + off, 4 + off, 5 + off, 6 + off
    for t in KNIGHT_T[k]:
        if b[t] == en:
            return True
    for t in KING_T[k]:
        if b[t] == ek:
            return True
    f = k & 7
    if white:
        if f > 0 and k + 7 < 64 and b[k + 7] == 7:
            return True
        if f < 7 and k + 9 < 64 and b[k + 9] == 7:
            return True
    else:
        if f < 7 and k - 7 >= 0 and b[k - 7] == 1:
            return True
        if f > 0 and k - 9 >= 0 and b[k - 9] == 1:
            return True
    rays = RAYS[k]
    for d in range(8):
        for t in rays[d]:
            c = b[t]
            if c:
                if c == eq or c == (er if d < 4 else eb):
                    return True
                break
    return False


def parse_chunk(data: bytes):
    """Worker: raw JSONL bytes -> (packed boards, stm, score, drop counters)."""
    cnt = collections.Counter()
    codes = bytearray()
    stms = bytearray()
    scores = []
    for line in data.split(b"\n"):
        if not line:
            continue
        cnt["lines"] += 1
        try:
            d = json.loads(line)
            ev = max(d["evals"], key=lambda e: e.get("depth", 0))
            pv = ev["pvs"][0]
            parts = d["fen"].split()
            b = [0] * 64
            r, f = 7, 0
            for ch in parts[0]:
                if ch == "/":
                    r -= 1
                    f = 0
                elif ch <= "9":
                    f += ord(ch) - 48
                else:
                    b[r * 8 + f] = CODE[ch]
                    f += 1
            white = parts[1] == "w"
        except Exception:
            cnt["malformed"] += 1
            continue
        if b.count(6) != 1 or b.count(12) != 1:
            cnt["malformed"] += 1
            continue
        if in_check(b, white):
            cnt["in_check"] += 1
            continue
        mv = pv.get("line", "").split(" ", 1)[0]
        if len(mv) < 4:
            cnt["no_move"] += 1
            continue
        if len(mv) == 5:
            cnt["promotion"] += 1
            continue
        fr = (ord(mv[1]) - 49) * 8 + ord(mv[0]) - 97
        to = (ord(mv[3]) - 49) * 8 + ord(mv[2]) - 97
        if not (0 <= fr < 64 and 0 <= to < 64):
            cnt["malformed"] += 1
            continue
        tc = b[to]
        if tc and (tc >= 7) == white:          # enemy piece on the target square
            cnt["capture"] += 1
            continue
        if b[fr] in (1, 7) and (fr & 7) != (to & 7) and tc == 0:
            cnt["capture"] += 1                 # en passant
            continue
        if "mate" in pv and pv["mate"] is not None:
            s = CLAMP if pv["mate"] > 0 else -CLAMP
        elif "cp" in pv and pv["cp"] is not None:
            s = max(-CLAMP, min(CLAMP, int(pv["cp"])))
        else:
            cnt["no_score"] += 1
            continue
        if abs(s) >= CLAMP_LEVEL:                # labels are White's point of view here
            edge = 0
            for c in b:
                if c:
                    edge += SIGN[c] * VAL[c]
            if (edge if s > 0 else -edge) < MIN_EDGE:
                cnt["clamp_level_material"] += 1
                continue
        if not white:
            s = -s
        codes += bytes(b)
        stms.append(0 if white else 1)
        scores.append(s)
        cnt["kept"] += 1
    a = np.frombuffer(bytes(codes), dtype=np.uint8).reshape(-1, 64)
    packed = (a[:, 0::2] | (a[:, 1::2] << 4)).astype(np.uint8)
    return packed, np.frombuffer(bytes(stms), dtype=np.uint8).copy(), \
        np.array(scores, dtype=np.int16), dict(cnt)


def open_stream(path):
    import zstandard
    if path:
        raw = open(path, "rb")
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "chess-nnue-parser"})
        raw = urllib.request.urlopen(req, timeout=120)
    return zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw, read_size=1 << 20)


def chunks(reader, skip):
    """Yield CHUNK_LINES-line blocks of raw bytes, after skipping `skip` lines."""
    rest = b""
    buf, n = [], 0
    while True:
        block = reader.read(8 << 20)
        if not block:
            break
        lines = (rest + block).split(b"\n")
        rest = lines.pop()
        if skip:
            drop = min(skip, len(lines))
            skip -= drop
            lines = lines[drop:]
        buf.extend(lines)
        n += len(lines)
        while n >= CHUNK_LINES:
            yield b"\n".join(buf[:CHUNK_LINES])
            buf = buf[CHUNK_LINES:]
            n -= CHUNK_LINES
    if rest:
        buf.append(rest)
    if buf:
        yield b"\n".join(buf)


class ShardWriter:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.index = len([f for f in os.listdir(out) if f.startswith("shard_")])
        self.parts = []
        self.n = 0
        self.total = 0
        self.sample = []                       # small sample for the sanity report

    def add(self, packed, stm, score):
        if len(score) == 0:
            return
        self.parts.append((packed, stm, score))
        self.n += len(score)
        self.total += len(score)
        if len(self.sample) < 50:
            self.sample.append((packed[:4000], stm[:4000], score[:4000]))
        if self.n >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        board = np.concatenate([p[0] for p in self.parts])
        stm = np.concatenate([p[1] for p in self.parts])
        score = np.concatenate([p[2] for p in self.parts])
        path = os.path.join(self.out, f"shard_{self.index:03d}.npz")
        np.savez(path, board=board, stm=stm, score=score)
        print(f"  wrote {path}  ({len(score):,} positions)", flush=True)
        self.index += 1
        self.parts, self.n = [], 0


def sanity(writer, counts, elapsed):
    kept = counts.get("kept", 0)
    lines = counts.get("lines", 0)
    print("\n================ SANITY REPORT ================")
    print(f"lines read          : {lines:,}")
    print(f"positions kept      : {kept:,}  ({100 * kept / max(1, lines):.1f}%)")
    for k in ("in_check", "capture", "promotion", "clamp_level_material", "no_score", "no_move", "malformed"):
        print(f"  dropped {k:22s}: {counts.get(k, 0):,}")
    print(f"time                : {elapsed / 60:.1f} min  ({lines / max(1, elapsed):,.0f} lines/s)")
    if not writer.sample:
        print("no sample -- nothing kept")
        return
    board = np.concatenate([s[0] for s in writer.sample])
    stm = np.concatenate([s[1] for s in writer.sample])
    score = np.concatenate([s[2] for s in writer.sample]).astype(np.float64)
    codes = np.empty((len(board), 64), dtype=np.uint8)
    codes[:, 0::2] = board & 15
    codes[:, 1::2] = board >> 4
    val = np.array(VAL, dtype=np.float64)[codes]
    sgn = np.array(SIGN, dtype=np.float64)[codes]
    edge = (val * sgn).sum(1) * np.where(stm == 0, 1, -1)      # side to move's material edge
    pieces = (codes > 0).sum(1)
    head = np.clip((pieces - 2) // 4, 0, 7)
    print(f"sample size         : {len(score):,}")
    print(f"mean |score|        : {np.abs(score).mean():.0f} cp;  at the clamp: {100 * np.mean(np.abs(score) >= CLAMP_LEVEL):.1f}%")
    print(f"corr(material edge, score) : {np.corrcoef(edge, score)[0, 1]:.3f}   (must be clearly positive)")
    up_q, dn_q, lvl = edge >= 8, edge <= -8, edge == 0
    print(f"mean score, side to move up >= a queen   : {score[up_q].mean() if up_q.any() else float('nan'):+.0f}  (must be well above +800)")
    print(f"mean score, side to move down >= a queen : {score[dn_q].mean() if dn_q.any() else float('nan'):+.0f}  (must be well below -800)")
    print(f"mean score, level material               : {score[lvl].mean() if lvl.any() else float('nan'):+.0f}  (expect a small positive number)")
    print("positions per material head (0 = fewest pieces):")
    print("   " + "  ".join(f"h{h}:{100 * np.mean(head == h):.1f}%" for h in range(8)))
    print("================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120_000_000, help="stop after this many kept positions")
    ap.add_argument("--minutes", type=float, default=80.0, help="stop after this many minutes")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="data_v3")
    ap.add_argument("--file", default="", help="local .jsonl.zst instead of downloading")
    ap.add_argument("--skip", type=int, default=0, help="skip this many lines first (to extend a finished run)")
    args = ap.parse_args()

    if os.path.exists("STOP"):
        os.remove("STOP")
    print(f"source  : {args.file or URL}")
    print(f"target  : {args.target:,} kept positions or {args.minutes:.0f} min;  workers {args.workers};  out {args.out}/")
    writer = ShardWriter(args.out)
    counts = collections.Counter()
    t0 = time.time()
    last = t0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers)
    pending = collections.deque()
    reason = "end of file"

    def collect(res):
        packed, stm, score, cnt = res
        counts.update(cnt)
        writer.add(packed, stm, score)

    try:
        for block in chunks(open_stream(args.file), args.skip):
            pending.append(pool.apply_async(parse_chunk, (block,)))
            while len(pending) >= 3 * args.workers:
                collect(pending.popleft().get())
            now = time.time()
            if now - last >= 15:
                last = now
                print(f"  {(now - t0) / 60:5.1f} min  lines {counts['lines']:>12,}  kept {writer.total:>12,}  "
                      f"({counts['lines'] / (now - t0):,.0f} lines/s)", flush=True)
            if writer.total >= args.target:
                reason = "target reached"
                break
            if now - t0 >= args.minutes * 60:
                reason = "time cap"
                break
            if os.path.exists("STOP"):
                reason = "STOP file"
                break
    except KeyboardInterrupt:
        reason = "Ctrl-C"
    except Exception as e:                      # network drop: keep what we have
        reason = f"stream error: {e!r}"
    finally:
        try:
            while pending:
                collect(pending.popleft().get())
        except KeyboardInterrupt:
            pass
        pool.terminate()
        writer.flush()
        print(f"\nstopped: {reason}")
        print(f"to extend this run later: python parse_v3.py --skip {counts['lines']}")
        sanity(writer, counts, time.time() - t0)
        print("DONE")


if __name__ == "__main__":
    main()