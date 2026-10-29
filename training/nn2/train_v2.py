"""Trainer v2 -- the network numbav9 will run.
 
    python train_v2.py --probe          # 30 seconds: measures speed, trains nothing useful
    python train_v2.py                  # the real run, 60 minute budget
    python train_v2.py --minutes 45     # a different budget
    python train_v2.py --resume         # continue from the last checkpoint
    python train_v2.py --steps 20000    # fixed length: same steps and batches on any hardware (--seed, default 0)
    python train_v2.py --steps 20000 --deterministic --cpu   # and bit for bit (one CPU thread; slow)
 
Stop early at any time with:  touch STOP   (the weights file is always usable)
 
The network
-----------
Input: 4 king buckets x 768 piece-square features = 3072, from EACH side's
point of view, with the same weights for both sides:
 
  * Perspective. Each side sees the board as if it were White: for Black
    the board is flipped vertically and the colours swapped. Whose move it
    is decides which accumulator goes first. This makes colour symmetry
    exact and removes the first network's +95 cp side-to-move artefact.
  * Horizontal mirror. If a side's king is on files e-h, that side's view
    is mirrored so its king is always on files a-d.
  * King buckets (by that side's own king, after the flip and mirror):
        0: back rank, files a-b   (castled short, or long to b1)
        1: back rank, files c-d   (uncastled, or long to c1)
        2: second rank
        3: third rank or further  (walked-out king, endgames)
  * Factoriser (training only). Each bucket's weights = a shared table + a
    bucket-specific table, so rare buckets learn from all the data. Merged
    into one table on export; costs nothing at runtime.
 
Layers: 3072 -> 256 (per perspective), SCReLU (clamp 0..1, then square),
concatenate own + opponent = 512 -> 1, with 8 output heads chosen by piece
count: head = min(7, (pieces - 2) // 4).
 
Loss: squared error in win-probability space, sigmoid(eval / 400), against
the Stockfish label. Large advantages still count but no longer dominate,
and the network is not rewarded for guessing close to zero.
 
Output files
------------
  nnue_v2.bin             weights for the gate and for numbav9 (current folder)
  <data-dir>/nnue_v2.ckpt checkpoint for --resume
"""
from __future__ import annotations
 
import argparse
import glob
import json
import math
import os
import queue
import struct
import threading
import time
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
PREFIX = "train_v2"
H = 256                 # accumulator width per perspective
NB = 4                  # king buckets
NH = 8                  # output heads
N_IN = NB * 768         # 3072 bucketed inputs
FAC_OFF = N_IN          # factoriser rows 3072..3839
PAD = N_IN + 768        # padding row 3840
EVAL_SCALE = 400.0      # network output 1.0 == 400 cp
W1_CLIP = 0.99          # table + factoriser merged stays inside +-1.98
W2_CLIP = 1.98
MAGIC = b"NNUEV2\x00\x00"
 
 
def bucket_of(sq: int) -> int:
    r, f = divmod(sq, 8)
    if r == 0:
        return 0 if f <= 1 else 1
    if r == 1:
        return 2
    return 3
 
 
BUCKET_TABLE = [bucket_of(sq) for sq in range(64)]
 
 
# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def make_indices(feat: torch.Tensor, bucket_t: torch.Tensor):
    """feat: (B, 32) int16, White-POV piece indices or -1.
 
    Returns (idx_white_view, idx_black_view), each (B, 64) long: 32 bucketed
    rows then 32 factoriser rows, padding = PAD. Also the output head (B,).
    """
    f = feat.long()
    valid = f >= 0
    f0 = f.clamp(min=0)
    col = f0 // 384
    pt = (f0 % 384) // 64
    sq = f0 % 64
    is_k = valid & (pt == 5)
    wk = (sq * (is_k & (col == 0))).sum(1)
    bk = (sq * (is_k & (col == 1))).sum(1)
 
    wflip = torch.where((wk % 8) >= 4, 7, 0)
    bflip = torch.where((bk % 8) >= 4, 7, 0) ^ 56
    wb = bucket_t[wk ^ wflip]
    bb = bucket_t[bk ^ bflip]
 
    sq_w = sq ^ wflip[:, None]
    sq_b = sq ^ bflip[:, None]
    base_w = col * 384 + pt * 64 + sq_w
    base_b = (1 - col) * 384 + pt * 64 + sq_b
    iw = torch.cat([wb[:, None] * 768 + base_w, FAC_OFF + base_w], 1)
    ib = torch.cat([bb[:, None] * 768 + base_b, FAC_OFF + base_b], 1)
    v2 = torch.cat([valid, valid], 1)
    iw = torch.where(v2, iw, PAD)
    ib = torch.where(v2, ib, PAD)
    head = ((valid.sum(1) - 2) // 4).clamp(0, NH - 1)
    return iw, ib, head
 
 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(PAD + 1, H, mode="sum", padding_idx=PAD)
        self.b1 = nn.Parameter(torch.full((H,), 0.5))
        self.w2 = nn.Parameter(torch.randn(NH, 2 * H) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(NH))
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 0.04)
            self.emb.weight[PAD].zero_()
        self.register_buffer("bucket_t", torch.tensor(BUCKET_TABLE, dtype=torch.long))
 
    def forward(self, feat, stm):
        """Returns the evaluation in network units, side to move's point of view."""
        iw, ib, head = make_indices(feat, self.bucket_t)
        B = feat.shape[0]
        acc = self.emb(torch.cat([iw, ib], 0)) + self.b1
        acc_w, acc_b = acc[:B], acc[B:]
        white = (stm == 0)[:, None]
        us = torch.where(white, acc_w, acc_b)
        them = torch.where(white, acc_b, acc_w)
        x = torch.cat([us, them], 1).clamp(0.0, 1.0).square()
        return (x * self.w2[head]).sum(1) + self.b2[head]
 
    @torch.no_grad()
    def clip_(self):
        self.emb.weight.clamp_(-W1_CLIP, W1_CLIP)
        self.w2.clamp_(-W2_CLIP, W2_CLIP)
 
    @torch.no_grad()
    def export(self, path: str, info: dict):
        w = self.emb.weight.detach().float().cpu().numpy()
        fac = w[FAC_OFF:FAC_OFF + 768]
        w1 = np.concatenate([w[b * 768:(b + 1) * 768] + fac for b in range(NB)], 0).astype("<f4")
        b1 = self.b1.detach().float().cpu().numpy().astype("<f4")
        w2 = self.w2.detach().float().cpu().numpy().astype("<f4")
        b2 = self.b2.detach().float().cpu().numpy().astype("<f4")
        meta = json.dumps(info).encode()
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(MAGIC)
            fh.write(struct.pack("<7i", 1, H, NB, NH, N_IN, int(EVAL_SCALE), len(meta)))
            fh.write(struct.pack("<64i", *BUCKET_TABLE))
            fh.write(meta)
            for a in (w1, b1, w2, b2):
                fh.write(a.tobytes())
        os.replace(tmp, path)
 
 
# --------------------------------------------------------------------------
# Benchmark (positions from your own games, Stockfish-labelled)
# --------------------------------------------------------------------------
_PIECE = {c: (0 if c.isupper() else 1, "pnbrqk".index(c.lower())) for c in "PNBRQKpnbrqk"}
 
 
def fen_to_feat(fen: str):
    parts = fen.split()
    idx = []
    rank, file_ = 7, 0
    for ch in parts[0]:
        if ch == "/":
            rank -= 1
            file_ = 0
        elif ch.isdigit():
            file_ += int(ch)
        else:
            col, pt = _PIECE[ch]
            idx.append(col * 384 + pt * 64 + rank * 8 + file_)
            file_ += 1
    idx.sort()
    return idx + [-1] * (32 - len(idx)), (0 if parts[1] == "w" else 1)
 
 
class Bench:
    def __init__(self, path, device):
        d = np.load(path)
        rows = [fen_to_feat(str(f)) for f in d["fen"]]
        self.feat = torch.tensor([r[0] for r in rows], dtype=torch.int16, device=device)
        self.stm = torch.tensor([r[1] for r in rows], dtype=torch.int8, device=device)
        self.sign = np.where(np.array([r[1] for r in rows]) == 0, 1.0, -1.0)
        self.sf = d["sf_cp"].astype(np.float64)
        k = np.abs(self.sf) < 1500
        self.quiet = d["quiet"] & k
        self.strict = d["strict"] & k
 
    @torch.no_grad()
    def score(self, net):
        net.eval()
        y = net(self.feat, self.stm).float().cpu().numpy() * EVAL_SCALE * self.sign
        net.train()
        q = np.corrcoef(y[self.quiet], self.sf[self.quiet])[0, 1]
        s = np.corrcoef(y[self.strict], self.sf[self.strict])[0, 1]
        return q, s
 
 
# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(data_dir, max_rows=0):
    """Load every shard into one pre-sized array (peak memory = the data, not twice it)."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{PREFIX}_feat_*.npy")))
    if not files:
        raise SystemExit(f"no training shards in {data_dir} -- run parse_v2.py first")
    sizes = []
    for fp in files:
        sizes.append(np.load(fp, mmap_mode="r").shape[0])
        if max_rows and sum(sizes) >= max_rows:
            break
    files = files[:len(sizes)]
    n = sum(sizes)
    print(f"loading  : {len(files)} shards, {n:,} positions, {n * 67 / 2 ** 30:.1f} GB", flush=True)
    feat = np.empty((n, 32), dtype=np.int16)
    stm = np.empty(n, dtype=np.int8)
    cp = np.empty(n, dtype=np.int16)
    o = 0
    for fp, sz in zip(files, sizes):
        i = fp[-7:-4]
        feat[o:o + sz] = np.load(fp)
        stm[o:o + sz] = np.load(os.path.join(data_dir, f"{PREFIX}_stm_{i}.npy"))
        cp[o:o + sz] = np.load(os.path.join(data_dir, f"{PREFIX}_cp_{i}.npy"))
        o += sz
    return feat, stm, cp
 
 
class Batches:
    """Random batches (with replacement). GPU-resident when it fits, otherwise
    gathered on the CPU in a background thread."""
 
    def __init__(self, feat, stm, cp, n_train, bsz, device):
        self.n, self.bsz, self.device = n_train, bsz, device
        need = n_train * (feat.shape[1] * 2 + 3)
        self.on_gpu = False
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            if need < free - 2.5 * 2 ** 30:
                self.on_gpu = True
        if self.on_gpu:
            self.f = torch.from_numpy(feat[:n_train]).to(device)
            self.s = torch.from_numpy(stm[:n_train]).to(device)
            self.c = torch.from_numpy(cp[:n_train]).to(device)
        else:
            self.f = torch.from_numpy(feat[:n_train])
            self.s = torch.from_numpy(stm[:n_train])
            self.c = torch.from_numpy(cp[:n_train])
            self.q = queue.Queue(maxsize=6)
            self.alive = True
            self.th = threading.Thread(target=self._fill, daemon=True)
            self.th.start()
 
    def _fill(self):
        pin = self.device.type == "cuda"
        while self.alive:
            i = torch.randint(self.n, (self.bsz,))
            b = (self.f.index_select(0, i), self.s.index_select(0, i), self.c.index_select(0, i))
            if pin:
                b = tuple(t.pin_memory() for t in b)
            self.q.put(b)
 
    def next(self):
        if self.on_gpu:
            i = torch.randint(self.n, (self.bsz,), device=self.device)
            return self.f[i], self.s[i], self.c[i]
        f, s, c = self.q.get()
        return (f.to(self.device, non_blocking=True), s.to(self.device, non_blocking=True),
                c.to(self.device, non_blocking=True))
 
    def close(self):
        if not self.on_gpu:
            self.alive = False
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
 
 
def loss_fn(y, stm, cp):
    target = cp.float() * (1.0 - 2.0 * stm.float()) / EVAL_SCALE
    return (torch.sigmoid(y) - torch.sigmoid(target)).square().mean()
 
 
@torch.no_grad()
def validate(net, vf, vs, vc, device):
    net.eval()
    tot_l, tot_mae, n = 0.0, 0.0, 0
    for i in range(0, len(vc), 32768):
        f = vf[i:i + 32768].to(device)
        s = vs[i:i + 32768].to(device)
        c = vc[i:i + 32768].to(device)
        y = net(f, s)
        tot_l += loss_fn(y, s, c).item() * len(c)
        tgt = c.float() * (1.0 - 2.0 * s.float())
        tot_mae += (y * EVAL_SCALE - tgt).abs().sum().item()
        n += len(c)
    net.train()
    return tot_l / n, tot_mae / n
 
 
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--bench", default="bench_v2.npz")
    ap.add_argument("--out", default="nnue_v2.bin")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="ignore the GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="one CPU thread and deterministic kernels, so a --steps run repeats bit for bit")
    ap.add_argument("--steps", type=int, default=0,
                    help="fixed number of steps; the default, 0, sets it from measured speed and --minutes, as run")
    args = ap.parse_args()
 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not (args.cpu or args.deterministic) else "cpu")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    elif device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    stop_file = os.path.abspath("STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    ckpt_path = os.path.join(args.data_dir, "nnue_v2.ckpt")
 
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU, {torch.get_num_threads()} threads"
    print(f"device   : {dev_name}", flush=True)
 
    t0 = time.time()
    meta_rows = 0
    mp = os.path.join(args.data_dir, f"{PREFIX}_meta.json")
    if os.path.exists(mp):
        meta_rows = json.load(open(mp)).get("rows", 0)
    feat, stm, cp = load_data(args.data_dir, max_rows=4_000_000 if args.probe else 0)
    n = len(cp)
    n_val = min(200_000, n // 20)
    n_train = n - n_val
    print(f"data     : {n:,} positions loaded in {time.time() - t0:.0f}s  "
          f"({n_train:,} train, {n_val:,} held out)", flush=True)
    vf = torch.from_numpy(feat[n_train:])
    vs = torch.from_numpy(stm[n_train:])
    vc = torch.from_numpy(cp[n_train:])
 
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    bench = Bench(args.bench, device)
    batches = Batches(feat, stm, cp, n_train, args.batch, device)
    print(f"batches  : {'GPU-resident' if batches.on_gpu else 'streamed from CPU memory'}, "
          f"{args.batch:,} positions each", flush=True)
 
    step = 0
    total_steps = None
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        step, total_steps = ck["step"], ck["total_steps"]
        print(f"resumed  : step {step:,} of {total_steps:,}", flush=True)
 
    q0, s0 = bench.score(net)
    print(f"bench before training: quiet {q0:.3f}  strict {s0:.3f}   "
          f"(to beat: v7 hand-written 0.868 / 0.905, v8 net 0.850 / 0.843)\n", flush=True)
 
    def lr_at(st):
        if total_steps is None:
            return args.lr * min(1.0, (st + 1) / 200)
        warm = min(1.0, (st + 1) / max(200, total_steps // 100))
        prog = min(1.0, st / total_steps)
        return args.lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
 
    def train_step():
        nonlocal step
        f, s, c = batches.next()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        y = net(f, s)
        loss = loss_fn(y, s, c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        net.clip_()
        step += 1
        return loss
 
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
 
    # ---- speed measurement (also the first training steps) -----------------
    probe_s = 30.0 if args.probe else 20.0
    if args.steps and not args.probe:
        sps = None                      # fixed schedule: no timed steps, so a rerun repeats step for step
    else:
        for _ in range(10):
            train_step()
        sync()
        tp = time.time()
        k0 = step
        while time.time() - tp < probe_s:
            train_step()
            if (step - k0) % 20 == 0:
                sync()
        sync()
        sps = (step - k0) / (time.time() - tp)
        pps = sps * args.batch
        print(f"speed    : {pps / 1e6:.2f} M positions/s  ({sps:.1f} steps/s)", flush=True)
 
    if args.probe:
        total = meta_rows or n
        for m in (30, 45, 60, 90):
            seen = pps * m * 60
            print(f"  {m:3d} min -> {seen / 1e6:7.0f} M positions = {seen / max(1, total):5.1f} passes over "
                  f"your {total / 1e6:.1f} M positions", flush=True)
        batches.close()
        print("DONE")
        return
 
    if total_steps is None:
        total_steps = args.steps or max(step + 100, int(sps * args.minutes * 60))
    about = f"about {args.minutes:.0f} min" if sps else "fixed by --steps"
    print(f"plan     : {total_steps:,} steps = {total_steps * args.batch / 1e6:,.0f} M positions "
          f"= {total_steps * args.batch / n_train:.1f} passes, {about}\n", flush=True)
    print("   min |     step | passes |  lr      | train loss | val loss | val MAE cp | bench quiet  strict", flush=True)
 
    ck_every = max(200, total_steps // 25)
    t_run = time.time()
    run_loss, run_n = 0.0, 0
    stop_reason = "finished"
    best = (-1.0, -1.0)
 
    def checkpoint(final=False):
        nonlocal best
        vl, vmae = validate(net, vf, vs, vc, device)
        q, s = bench.score(net)
        best = max(best, (s, q))
        info = {"step": step, "total_steps": total_steps, "bench_quiet": round(q, 4), "bench_strict": round(s, 4),
                "val_mae_cp": round(vmae, 1), "positions": n_train, "final": final}
        net.export(args.out, info)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "total_steps": total_steps},
                   ckpt_path)
        el = (time.time() - t_run) / 60
        tl = run_loss / max(1, run_n)
        print(f" {el:5.1f} | {step:8,} | {step * args.batch / n_train:6.2f} | {lr_at(step):.2e} | "
              f"{tl:10.6f} | {vl:8.6f} | {vmae:10.1f} | {q:10.3f} {s:7.3f}", flush=True)
 
    try:
        while step < total_steps:
            loss = train_step()
            if step % 50 == 0:
                run_loss += loss.item()
                run_n += 1
            if step % ck_every == 0:
                checkpoint()
                run_loss, run_n = 0.0, 0
            if step % 100 == 0 and os.path.exists(stop_file):
                stop_reason = "STOP file"
                break
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    checkpoint(final=True)
    batches.close()
    if os.path.exists(stop_file):
        os.remove(stop_file)
    print(f"\nstopped  : {stop_reason} after {(time.time() - t_run) / 60:.1f} min", flush=True)
    print(f"weights  : {os.path.abspath(args.out)}", flush=True)
    print("next     : python gate_v2.py", flush=True)
    print("DONE")
 
 
if __name__ == "__main__":
    main()
