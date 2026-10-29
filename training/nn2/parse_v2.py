"""Parser v2 -- stream the Lichess evaluation database into NNUE training shards.
 
    python parse_v2.py                      # stream from lichess.org, 35 minute cap
    python parse_v2.py --minutes 20         # shorter cap
    python parse_v2.py --resume             # carry on after a dropped connection
    python parse_v2.py --input FILE.zst     # read a local copy instead of the network
 
Stop early at any time with:  touch STOP   (in the folder you ran it from)
Everything written so far is kept. Ctrl+C also works, but Git Bash sometimes
kills Python outright on Ctrl+C, so STOP is the reliable way.
 
What changed from the first parser, and why
-------------------------------------------
1. QUIET FILTER. A position is dropped when the best move in the deepest
   evaluation is a capture or a promotion. The engine only ever asks the
   network about positions that quiescence has already settled; the first
   network was trained on unsettled ones and learned to discount material.
2. MATES AND BIG SCORES ARE KEPT, clamped to +-CP_CLIP. Dropping them hid
   every clearly-won position from the network, which is why it scored
   queen-against-bare-king at +444.
3. Side-to-move-in-check positions are still dropped (search never
   evaluates those statically), as are malformed rows.
4. Positions are unique by construction: the Lichess file has one row per FEN.
 
Output (in --data-dir, default data/):
 
    train_v2_feat_NNN.npy  int16 (n, 32)  piece features, -1 padded
    train_v2_stm_NNN.npy   int8  (n,)     side to move: 0 white, 1 black
    train_v2_cp_NNN.npy    int16 (n,)     Stockfish score, WHITE's point of view
    train_v2_meta.json     progress, used by --resume and by the trainer
 
Feature index (White's point of view; the trainer derives both perspectives):
 
    index = colour * 384 + (piece_type - 1) * 64 + square
    colour 0 = white; piece_type 1..6 = P N B R Q K; square 0 = a1 .. 63 = h8
"""
from __future__ import annotations
 
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import urllib.request
from collections import deque
 
import numpy as np
 
URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
CP_CLIP = 2000          # scores and mates are clamped to +-CP_CLIP
MIN_DEPTH = 16          # ignore evaluations shallower than this
CHUNK_LINES = 20_000    # lines per worker task
SHARD_ROWS = 2_000_000  # kept positions per shard file
PREFIX = "train_v2"
 
# --------------------------------------------------------------------------
# Board helpers (plain Python: they run inside the worker processes)
# --------------------------------------------------------------------------
_PIECE = {c: (0 if c.isupper() else 1, "pnbrqk".index(c.lower()))
          for c in "PNBRQKpnbrqk"}
 
 
def _targets(deltas):
    out = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        lst = []
        for dr, df in deltas:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                lst.append(rr * 8 + ff)
        out.append(tuple(lst))
    return tuple(out)
 
 
KNIGHT_T = _targets([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
KING_T = _targets([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
 
 
def _rays(dirs):
    out = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        per = []
        for dr, df in dirs:
            ray = []
            rr, ff = r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                ray.append(rr * 8 + ff)
                rr += dr
                ff += df
            if ray:
                per.append(tuple(ray))
        out.append(tuple(per))
    return tuple(out)
 
 
ORTHO = _rays([(1, 0), (-1, 0), (0, 1), (0, -1)])
DIAG = _rays([(1, 1), (1, -1), (-1, 1), (-1, -1)])
# piece code = colour * 6 + type_index   (type_index 0..5 = P N B R Q K)
 
 
def in_check(board, stm):
    """Is the side to move's king attacked? board: list of 64 codes, -1 empty."""
    king = stm * 6 + 5
    ks = board.index(king)
    opp = 1 - stm
    o = opp * 6
    r, f = divmod(ks, 8)
    # pawns
    if stm == 0:      # white king, black pawns attack from rank r+1
        if r < 7:
            if f > 0 and board[ks + 7] == o + 0:
                return True
            if f < 7 and board[ks + 9] == o + 0:
                return True
    else:             # black king, white pawns attack from rank r-1
        if r > 0:
            if f > 0 and board[ks - 9] == o + 0:
                return True
            if f < 7 and board[ks - 7] == o + 0:
                return True
    for t in KNIGHT_T[ks]:
        if board[t] == o + 1:
            return True
    for t in KING_T[ks]:
        if board[t] == o + 5:
            return True
    rq = (o + 3, o + 4)
    for ray in ORTHO[ks]:
        for t in ray:
            p = board[t]
            if p != -1:
                if p in rq:
                    return True
                break
    bq = (o + 2, o + 4)
    for ray in DIAG[ks]:
        for t in ray:
            p = board[t]
            if p != -1:
                if p in bq:
                    return True
                break
    return False
 
 
def _sq(s):
    return (ord(s[1]) - 49) * 8 + (ord(s[0]) - 97)
 
 
# drop / tally reasons, in this order
R_KEPT, R_MALFORMED, R_SHALLOW, R_CHECK, R_CAPTURE, R_PROMO, R_MATE_KEPT, R_CLIPPED = range(8)
N_REASONS = 8
 
 
def parse_chunk(lines, min_depth):
    """Worker: bytes lines -> (feat, stm, cp, tallies)."""
    n = len(lines)
    feat = np.full((n, 32), -1, dtype=np.int16)
    stm_a = np.zeros(n, dtype=np.int8)
    cp_a = np.zeros(n, dtype=np.int16)
    tally = np.zeros(N_REASONS, dtype=np.int64)
    k = 0
    loads = json.loads
    for raw in lines:
        try:
            d = loads(raw)
            fen = d["fen"]
            evals = d["evals"]
            best = None
            for e in evals:
                if best is None or (e.get("depth", 0), e.get("knodes", 0)) > (best.get("depth", 0), best.get("knodes", 0)):
                    best = e
            if best is None or not best.get("pvs"):
                tally[R_MALFORMED] += 1
                continue
            if best.get("depth", 0) < min_depth:
                tally[R_SHALLOW] += 1
                continue
            pv = best["pvs"][0]
            line = pv.get("line", "")
            if not line:
                tally[R_MALFORMED] += 1
                continue
            mv = line.split(" ", 1)[0]
            if "mate" in pv:
                cp = CP_CLIP if pv["mate"] > 0 else -CP_CLIP
                is_mate = True
            else:
                cp = int(pv["cp"])
                is_mate = False
            parts = fen.split(" ")
            placement, side = parts[0], parts[1]
            stm = 0 if side == "w" else 1
            board = [-1] * 64
            idx = []
            rank, file_ = 7, 0
            for ch in placement:
                if ch == "/":
                    rank -= 1
                    file_ = 0
                elif ch <= "9":
                    file_ += ord(ch) - 48
                else:
                    col, pt = _PIECE[ch]
                    sq = rank * 8 + file_
                    board[sq] = col * 6 + pt
                    idx.append(col * 384 + pt * 64 + sq)
                    file_ += 1
            if len(idx) > 32 or board.count(5) != 1 or board.count(11) != 1:
                tally[R_MALFORMED] += 1
                continue
            if in_check(board, stm):
                tally[R_CHECK] += 1
                continue
            if len(mv) == 5:
                tally[R_PROMO] += 1
                continue
            fr, to = _sq(mv[0:2]), _sq(mv[2:4])
            mover, target = board[fr], board[to]
            if mover == -1 or mover // 6 != stm:
                tally[R_MALFORMED] += 1
                continue
            if target != -1 and target // 6 != stm:
                tally[R_CAPTURE] += 1
                continue
            if mover % 6 == 0 and (fr & 7) != (to & 7) and target == -1:   # en passant
                tally[R_CAPTURE] += 1
                continue
            if cp > CP_CLIP:
                cp = CP_CLIP
                tally[R_CLIPPED] += 1
            elif cp < -CP_CLIP:
                cp = -CP_CLIP
                tally[R_CLIPPED] += 1
            if is_mate:
                tally[R_MATE_KEPT] += 1
            idx.sort()
            feat[k, :len(idx)] = idx
            stm_a[k] = stm
            cp_a[k] = cp
            k += 1
            tally[R_KEPT] += 1
        except Exception:
            tally[R_MALFORMED] += 1
    return feat[:k], stm_a[:k], cp_a[:k], tally
 
 
def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
 
# --------------------------------------------------------------------------
# Input stream
# --------------------------------------------------------------------------
class CountingReader(io.RawIOBase):
    def __init__(self, f):
        self.f = f
        self.count = 0
 
    def readable(self):
        return True
 
    def readinto(self, b):
        data = self.f.read(len(b))
        n = len(data)
        b[:n] = data
        self.count += n
        return n
 
 
def open_stream(args):
    import zstandard
    if args.input:
        raw = open(args.input, "rb")
        total = os.path.getsize(args.input)
    else:
        req = urllib.request.Request(args.url, headers={"User-Agent": "chess-nnue-parser"})
        raw = urllib.request.urlopen(req, timeout=60)
        total = int(raw.headers.get("Content-Length", 0))
    counter = CountingReader(raw)
    if args.input and not args.input.endswith(".zst"):
        stream = io.BufferedReader(counter, buffer_size=1 << 20)
    else:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        stream = io.BufferedReader(dctx.stream_reader(io.BufferedReader(counter, 1 << 20), read_size=1 << 20),
                                   buffer_size=1 << 20)
    return stream, counter, total
 
 
# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------
def shard_path(data_dir, kind, i):
    return os.path.join(data_dir, f"{PREFIX}_{kind}_{i:03d}.npy")
 
 
def write_shard(data_dir, i, feats, stms, cps):
    f = np.concatenate(feats)
    s = np.concatenate(stms)
    c = np.concatenate(cps)
    for kind, arr in (("feat", f), ("stm", s), ("cp", c)):
        tmp = shard_path(data_dir, kind, i) + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, shard_path(data_dir, kind, i))
    return len(c)
 
 
def save_meta(data_dir, meta):
    tmp = os.path.join(data_dir, f"{PREFIX}_meta.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, os.path.join(data_dir, f"{PREFIX}_meta.json"))
 
 
# --------------------------------------------------------------------------
# Sanity report
# --------------------------------------------------------------------------
VAL = np.array([1, 3, 3, 5, 9, 0], dtype=np.int32)
 
 
def sanity(data_dir, meta):
    n_sh = meta["shards"]
    tot = 0
    pcs_min, pcs_max, pcs_sum = 99, 0, 0
    fmin, fmax = 10 ** 9, -1
    cp_all = []
    stm_w = 0
    buckets = {k: [0, 0.0] for k in ("w_up8", "b_up8", "w_up3", "b_up3", "level")}
    for i in range(n_sh):
        f = np.load(shard_path(data_dir, "feat", i), mmap_mode="r")
        s = np.load(shard_path(data_dir, "stm", i))
        c = np.load(shard_path(data_dir, "cp", i)).astype(np.int64)
        f = np.asarray(f)
        valid = f >= 0
        cnt = valid.sum(1)
        pcs_min, pcs_max, pcs_sum = min(pcs_min, cnt.min()), max(pcs_max, cnt.max()), pcs_sum + cnt.sum()
        fmin, fmax = min(fmin, f[valid].min()), max(fmax, f[valid].max())
        fc = np.where(valid, f, 0).astype(np.int32)
        col = fc // 384
        pt = (fc % 384) // 64
        mat = (np.where(valid, VAL[pt] * (1 - 2 * col), 0)).sum(1)
        for key, m in (("w_up8", mat >= 8), ("b_up8", mat <= -8), ("w_up3", (mat >= 3) & (mat <= 4)),
                       ("b_up3", (mat <= -3) & (mat >= -4)), ("level", mat == 0)):
            buckets[key][0] += int(m.sum())
            buckets[key][1] += float(c[m].sum())
        stm_w += int((s == 0).sum())
        cp_all.append(c[:: max(1, len(c) // 200_000)])
        tot += len(c)
    cp_s = np.concatenate(cp_all)
    t = meta["tally"]
    lines = meta["lines_done"]
 
    def pct(x):
        return f"{100 * x / max(1, lines):5.1f}%"
 
    print("\n" + "=" * 64)
    print("PARSE REPORT")
    print("=" * 64)
    print(f"  lines read          : {lines:,}")
    print(f"  kept                : {t[R_KEPT]:,}  ({pct(t[R_KEPT])})")
    print(f"  dropped  capture    : {t[R_CAPTURE]:,}  ({pct(t[R_CAPTURE])})   <- the new quiet filter")
    print(f"           promotion  : {t[R_PROMO]:,}  ({pct(t[R_PROMO])})")
    print(f"           in check   : {t[R_CHECK]:,}  ({pct(t[R_CHECK])})")
    print(f"           shallow    : {t[R_SHALLOW]:,}  ({pct(t[R_SHALLOW])})")
    print(f"           malformed  : {t[R_MALFORMED]:,}  ({pct(t[R_MALFORMED])})")
    print(f"  mates kept (as +-{CP_CLIP}): {t[R_MATE_KEPT]:,}   scores clamped: {t[R_CLIPPED]:,}")
    print(f"  shards              : {n_sh}  ({tot:,} positions)")
    print("\nsanity")
    print(f"  pieces per position : min {pcs_min}, max {pcs_max}, mean {pcs_sum / max(1, tot):.1f}   (expect 2 .. 32)")
    print(f"  feature index range : {fmin} .. {fmax}   (must be inside 0 .. 767)")
    print(f"  score (White POV)   : mean {cp_s.mean():+.0f}, median {np.median(cp_s):+.0f}, sd {cp_s.std():.0f}")
    print(f"  white to move       : {100 * stm_w / max(1, tot):.1f}%")
 
    def mean(key):
        n, sm = buckets[key]
        return (sm / n if n else float("nan")), n
 
    m8w, n8w = mean("w_up8")
    m8b, n8b = mean("b_up8")
    m3w, n3w = mean("w_up3")
    m3b, n3b = mean("b_up3")
    m0, n0 = mean("level")
    print("\n  mean score by material balance (quiet positions, so the balance is real):")
    print(f"    White up >= 8 pawns of material : {m8w:+7.0f}   ({n8w:,} positions)   want > +600 (was +344)")
    print(f"    Black up >= 8 pawns of material : {m8b:+7.0f}   ({n8b:,} positions)   want < -600")
    print(f"    White up a minor piece (3-4)    : {m3w:+7.0f}   ({n3w:,})")
    print(f"    Black up a minor piece (3-4)    : {m3b:+7.0f}   ({n3b:,})")
    print(f"    material level                  : {m0:+7.0f}   ({n0:,})")
    ok = (pcs_min >= 2 and pcs_max <= 32 and fmin >= 0 and fmax <= 767 and m8w > 600 and m8b < -600 and m8w > m3w
          and m3w > m0 > m3b and m3b > m8b and tot > 0)
    print()
    if ok:
        print("  VERDICT: OK ")
    else:
        print("  VERDICT: STOP -- something is off.")
    print("DONE")
 
 
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--input", default="", help="local .zst or .jsonl instead of the network")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--minutes", type=float, default=35.0, help="stop after this long (default 35)")
    ap.add_argument("--max-lines", type=int, default=0, help="stop after this many lines (0 = no limit)")
    ap.add_argument("--min-depth", type=int, default=MIN_DEPTH)
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--resume", action="store_true", help="skip lines already parsed and append")
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS, help=argparse.SUPPRESS)
    ap.add_argument("--report-only", action="store_true", help="just print the report for existing shards")
    args = ap.parse_args()
 
    os.makedirs(args.data_dir, exist_ok=True)
    meta_path = os.path.join(args.data_dir, f"{PREFIX}_meta.json")
    stop_file = os.path.abspath("STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
 
    if args.report_only:
        sanity(args.data_dir, json.load(open(meta_path)))
        return
 
    if args.resume and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"resuming: {meta['lines_done']:,} lines and {meta['shards']} shards already done", flush=True)
    else:
        for p in glob.glob(os.path.join(args.data_dir, f"{PREFIX}_*")):
            os.remove(p)
        meta = {"lines_done": 0, "shards": 0, "rows": 0, "tally": [0] * N_REASONS,
                "cp_clip": CP_CLIP, "min_depth": args.min_depth, "source": args.input or args.url}
        save_meta(args.data_dir, meta)
 
    print(f"data dir : {os.path.abspath(args.data_dir)}", flush=True)
    print(f"source   : {args.input or args.url}", flush=True)
    print(f"workers  : {args.workers}   time cap {args.minutes:.0f} min"
          + (f"   line cap {args.max_lines:,}" if args.max_lines else ""), flush=True)
    print("stop early any time with:  touch STOP\n", flush=True)
 
    stream, counter, total_bytes = open_stream(args)
    skip = meta["lines_done"]
    if skip:
        t0 = time.time()
        for i, _ in enumerate(stream, 1):
            if i >= skip:
                break
            if i % 5_000_000 == 0:
                print(f"  skipping already-parsed lines: {i:,} / {skip:,}", flush=True)
        print(f"  skipped {skip:,} lines in {time.time() - t0:.0f}s", flush=True)
 
    pool = mp.Pool(args.workers, initializer=_worker_init)
    pending = deque()
    buf_f, buf_s, buf_c, buf_rows, buf_lines = [], [], [], 0, 0
    lines_read = 0          # lines handed to workers this run
    t_start = time.time()
    t_last = 0.0
    start_done = meta["lines_done"]
    stop_reason = "end of file"
 
    def collect(block):
        nonlocal buf_rows, buf_lines
        res, nlines = pending.popleft()
        f, s, c, tally = res.get() if block else res.get(timeout=600)
        buf_f.append(f)
        buf_s.append(s)
        buf_c.append(c)
        buf_rows += len(c)
        buf_lines += nlines
        for j in range(N_REASONS):
            meta["tally"][j] += int(tally[j])
        if buf_rows >= args.shard_rows:
            flush_shard()
 
    def flush_shard():
        nonlocal buf_f, buf_s, buf_c, buf_rows, buf_lines
        if buf_rows:
            write_shard(args.data_dir, meta["shards"], buf_f, buf_s, buf_c)
            meta["shards"] += 1
            meta["rows"] += buf_rows
        meta["lines_done"] += buf_lines
        save_meta(args.data_dir, meta)
        buf_f, buf_s, buf_c, buf_rows, buf_lines = [], [], [], 0, 0
 
    def progress(force=False):
        nonlocal t_last
        now = time.time()
        if not force and now - t_last < 10:
            return
        t_last = now
        el = now - t_start
        done = meta["lines_done"] + buf_lines
        kept = meta["rows"] + buf_rows
        rate = (done - start_done) / max(el, 1e-9)
        mb = counter.count / 2 ** 20
        frac = f"  {100 * counter.count / total_bytes:4.1f}% of file" if total_bytes else ""
        left = max(0.0, args.minutes * 60 - el)
        print(f"  {el / 60:5.1f} min | lines {done:>12,} | kept {kept:>11,} | {rate / 1000:6.1f}k lines/s | "
              f"{mb:8.0f} MB in{frac} | cap in {left / 60:4.1f} min", flush=True)
 
    try:
        chunk = []
        for raw in stream:
            chunk.append(raw)
            if len(chunk) >= CHUNK_LINES:
                pending.append((pool.apply_async(parse_chunk, (chunk, args.min_depth)), len(chunk)))
                lines_read += len(chunk)
                chunk = []
                while len(pending) >= 2 * args.workers:
                    collect(True)
                progress()
                if args.max_lines and meta["lines_done"] + buf_lines + len(pending) * CHUNK_LINES >= args.max_lines:
                    stop_reason = "line cap"
                    break
                if time.time() - t_start >= args.minutes * 60:
                    stop_reason = "time cap"
                    break
                if os.path.exists(stop_file):
                    stop_reason = "STOP file"
                    break
        else:
            if chunk:
                pending.append((pool.apply_async(parse_chunk, (chunk, args.min_depth)), len(chunk)))
                lines_read += len(chunk)
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    except Exception as exc:     # dropped connection, truncated stream, ...
        stop_reason = f"stream error: {exc!r}  (run again with --resume to continue)"
 
    print(f"\nstopping: {stop_reason}. finishing the chunks already in flight...", flush=True)
    try:
        while pending:
            collect(False)
    except Exception as exc:
        print(f"  (lost the in-flight chunks: {exc!r})", flush=True)
        pending.clear()
    flush_shard()
    pool.terminate()
    progress(force=True)
    if os.path.exists(stop_file):
        os.remove(stop_file)
    sanity(args.data_dir, meta)
 
 
if __name__ == "__main__":
    mp.freeze_support()
    main()
 
