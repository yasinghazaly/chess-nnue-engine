"""train_v3.py -- train NN3, the final evaluation network, for numbav10.

    python train_v3.py --hours 5        # the overnight run
    touch STOP                          # end early; the best network is already saved

Reads data_v3/shard_*.npz (from parse_v3.py). Put numbav9.py in the same
folder and NN2 is scored on the same held-out positions for comparison.

Network (same form as NN2, so the engine kernel only changes constants):
    two perspectives, each (8 king buckets x 768) -> 384, SCReLU, 8 material heads
    king bucket of the (flipped, mirrored to files a-d) king square:
        rank 1: a-b | c | d        rank 2: a-b | c-d        rank 3 | rank 4 | rank 5+
    a shared 768-row "factoriser" is added to every bucket during training and
    merged into the buckets at export, so rarely-seen buckets still learn.
Loss: run D -- (sigmoid(pred/800) - sigmoid(target/800))^2, target = Stockfish cp.
Schedule: learning rate warms up, then cosine-decays to 1% exactly at --hours.

Output (rewritten every time the held-out loss improves):
    nnue_v3.bin   int16 W1 | int16 B1 | int16 W2 | int32 B2, little-endian,
                  the same layout and quantisation as NN2 (QA 510, QB 4096)
    nnue_v3.pt    float weights of the same checkpoint
"""
from __future__ import annotations

import argparse
import base64
import glob
import math
import os
import re
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

NB, H, NHEAD = 8, 384, 8
FAC = NB * 768                  # first factoriser row
PAD = FAC + 768                 # all-zero padding row
QA, QB = 510, 4096
K = 800.0
VAL_N = 1_000_000


def bucket_table(nb):
    t = np.zeros(64, dtype=np.int64)
    for s in range(64):
        r, f = s >> 3, s & 7            # f <= 3 after mirroring
        if nb == 8:
            if r == 0:
                t[s] = 0 if f <= 1 else (1 if f == 2 else 2)
            elif r == 1:
                t[s] = 3 if f <= 1 else 4
            else:
                t[s] = min(r + 3, 7)    # rank 3 -> 5, rank 4 -> 6, rank 5+ -> 7
        else:                           # NN2's four buckets, for the comparison
            t[s] = (0 if f <= 1 else 1) if r == 0 else (2 if r == 1 else 3)
    return t


def make_idx(board, stm, table, nb, factor):
    """uint8 (B, 32) packed boards -> feature indices exactly as the engine builds them.

    White's view: bucket*768 + colour*384 + piece*64 + (sq ^ wfl)
    Black's view: bucket*768 + (1-colour)*384 + piece*64 + (sq ^ bfl), bfl includes ^56
    """
    B = board.shape[0]
    dev = board.device
    b = board.long()
    codes = torch.stack((b & 15, b >> 4), dim=2).reshape(B, 64)
    occ = codes > 0
    dest = torch.where(occ, occ.cumsum(1) - 1, torch.full_like(codes, 32))
    sqs = torch.arange(64, device=dev).expand(B, 64)
    sq = torch.zeros(B, 33, dtype=torch.long, device=dev).scatter_(1, dest, sqs)[:, :32]
    c = torch.zeros(B, 33, dtype=torch.long, device=dev).scatter_(1, dest, codes)[:, :32]
    valid = c > 0
    col = torch.div(c - 1, 6, rounding_mode="floor").clamp(0, 1)
    pt = (c - 1) % 6
    wk = (codes == 6).long().argmax(1)
    bk = (codes == 12).long().argmax(1)
    wfl = torch.where((wk & 7) >= 4, 7, 0)
    bfl = torch.where((bk & 7) >= 4, 7, 0) ^ 56
    wb = table[wk ^ wfl]
    bb = table[bk ^ bfl]
    fw = col * 384 + pt * 64 + (sq ^ wfl[:, None])
    fb = (1 - col) * 384 + pt * 64 + (sq ^ bfl[:, None])
    pad = nb * 768 + (768 if factor else 0)
    iw = torch.where(valid, wb[:, None] * 768 + fw, pad)
    ib = torch.where(valid, bb[:, None] * 768 + fb, pad)
    if factor:
        iw = torch.cat((iw, torch.where(valid, nb * 768 + fw, pad)), 1)
        ib = torch.cat((ib, torch.where(valid, nb * 768 + fb, pad)), 1)
    head = torch.div(valid.sum(1) - 2, 4, rounding_mode="floor").clamp(0, NHEAD - 1)
    return iw, ib, head


def forward(w1, b1, w2, b2, pad, iw, ib, stm, head):
    """Float network output (multiply by 400 for centipawns, side to move's view)."""
    aw = F.embedding_bag(iw, w1, mode="sum", padding_idx=pad) + b1
    ab = F.embedding_bag(ib, w1, mode="sum", padding_idx=pad) + b1
    white = (stm == 0)[:, None]
    x = torch.cat((torch.where(white, aw, ab), torch.where(white, ab, aw)), 1)
    x = x.clamp(0.0, 1.0).square()
    return (x * w2[head]).sum(1) + b2[head]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.randn(PAD + 1, H) * 0.03
        w[PAD] = 0.0
        self.w1 = torch.nn.Parameter(w)
        self.b1 = torch.nn.Parameter(torch.full((H,), 0.5))
        self.w2 = torch.nn.Parameter(torch.randn(NHEAD, 2 * H) * 0.05)
        self.b2 = torch.nn.Parameter(torch.zeros(NHEAD))

    def forward(self, iw, ib, stm, head):
        return forward(self.w1, self.b1, self.w2, self.b2, PAD, iw, ib, stm, head)

    @torch.no_grad()
    def clip(self):
        # int16/int32 headroom the engine's loader checks:
        # |B1| + 32 |W1| < 32767 with merged |W1| <= 1.9 and |B1| <= 1.5 (x510)
        self.w1.clamp_(-0.95, 0.95)
        self.b1.clamp_(-1.5, 1.5)
        self.w2.clamp_(-1.98, 1.98)          # x4096 stays below 8256

    @torch.no_grad()
    def merged(self):
        w = self.w1[:FAC].reshape(NB, 768, H) + self.w1[FAC:PAD][None]
        return w.reshape(FAC, H)


def export(net, path):
    w1 = np.round(net.merged().cpu().numpy() * QA).astype("<i2")
    b1 = np.round(net.b1.detach().cpu().numpy() * QA).astype("<i2")
    w2 = np.round(net.w2.detach().cpu().numpy() * QB).astype("<i2")
    b2 = np.round(net.b2.detach().cpu().numpy() * QA * QB).astype("<i4")
    worst = int(np.abs(b1).max()) + 32 * int(np.abs(w1).max())
    assert worst < 32767 and int(np.abs(w2).max()) < 8256, "quantisation headroom"
    raw = w1.tobytes() + b1.tobytes() + w2.tobytes() + b2.tobytes()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    return w1, b1, w2, b2, worst, zlib.crc32(raw)


def eval_int(w1, b1, w2, b2, table, nb, board, stm):
    """Integer forward pass written exactly like the engine kernel (numpy, one position)."""
    iw, ib, head = make_idx(torch.from_numpy(board[None]), torch.from_numpy(np.array([stm])),
                            torch.from_numpy(table), nb, False)
    aw = b1.astype(np.int64).copy()
    ab = b1.astype(np.int64).copy()
    for k in iw[0].tolist():
        if k < nb * 768:
            aw += w1[k]
    for k in ib[0].tolist():
        if k < nb * 768:
            ab += w1[k]
    h = int(head[0])
    us, them = (aw, ab) if stm == 0 else (ab, aw)
    hh = len(b1)
    s = int((np.clip(us, 0, QA) ** 2 * w2[h, :hh].astype(np.int64)).sum()) + \
        int((np.clip(them, 0, QA) ** 2 * w2[h, hh:].astype(np.int64)).sum())
    y = s // QA + int(b2[h])
    return (y * 400) // (QA * QB)


def load_nn2(path, dev):
    src = open(path, encoding="utf-8", errors="ignore").read()
    blob = re.search(r'^_NNUE2_BLOB = "([^"]+)"', src, re.M).group(1)
    raw = zlib.decompress(base64.b64decode(blob))
    n1, n2, n3 = 4 * 768 * 256, 256, 8 * 512
    w1 = np.frombuffer(raw, "<i2", n1, 0).reshape(3072, 256)
    b1 = np.frombuffer(raw, "<i2", n2, 2 * n1)
    w2 = np.frombuffer(raw, "<i2", n3, 2 * (n1 + n2)).reshape(8, 512)
    b2 = np.frombuffer(raw, "<i4", 8, 2 * (n1 + n2 + n3))
    w1f = torch.zeros(3072 + 1, 256)
    w1f[:3072] = torch.from_numpy(w1.astype(np.float32) / QA)
    return (w1f.to(dev), torch.from_numpy(b1.astype(np.float32) / QA).to(dev),
            torch.from_numpy(w2.astype(np.float32) / QB).to(dev),
            torch.from_numpy(b2.astype(np.float32) / (QA * QB)).to(dev))


def load_data():
    files = sorted(glob.glob(os.path.join("data_v3", "shard_*.npz")))
    assert files, "no data_v3/shard_*.npz -- run parse_v3.py first"
    sizes = [np.load(f, mmap_mode="r")["score"].shape[0] for f in files]
    n = sum(sizes)
    board = np.empty((n, 32), np.uint8)
    stm = np.empty(n, np.uint8)
    score = np.empty(n, np.int16)
    o = 0
    for f, s in zip(files, sizes):
        d = np.load(f)
        board[o:o + s], stm[o:o + s], score[o:o + s] = d["board"], d["stm"], d["score"]
        o += s
    print(f"data    : {n:,} positions from {len(files)} shards")
    return board, stm, score


@torch.no_grad()
def evaluate(fn, vb, vs, vy, table, nb, factor):
    """Held-out loss (run D), and MAE in cp on |Stockfish| < 1500."""
    loss, mae, nmae, preds = 0.0, 0.0, 0, []
    for i in range(0, len(vy), 65536):
        b, s, y = vb[i:i + 65536], vs[i:i + 65536], vy[i:i + 65536].float()
        iw, ib, head = make_idx(b, s, table, nb, factor)
        p = fn(iw, ib, s, head) * 400
        loss += ((torch.sigmoid(p / K) - torch.sigmoid(y / K)) ** 2).sum().item()
        m = y.abs() < 1500
        mae += (p[m] - y[m]).abs().sum().item()
        nmae += int(m.sum())
        preds.append(p)
    return loss / len(vy), mae / max(1, nmae), torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-min", type=float, default=10.0, help="minutes between held-out checks")
    args = ap.parse_args()
    if os.path.exists("STOP"):
        os.remove("STOP")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device  : {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}")
    board, stm, score = load_data()
    n_train = len(score) - VAL_N
    assert n_train > 0, "need more than 1M positions"
    gpu_data = False
    if dev.type == "cuda":
        free = torch.cuda.mem_get_info()[0]
        gpu_data = board.nbytes + stm.nbytes + score.nbytes + (2 << 30) < free
    ddev = dev if gpu_data else torch.device("cpu")
    tb = torch.from_numpy(board).to(ddev)
    ts = torch.from_numpy(stm).to(ddev)
    ty = torch.from_numpy(score).to(ddev)
    del board, stm, score
    print(f"storage : training data on {'GPU' if gpu_data else 'CPU'}; {n_train:,} train, {VAL_N:,} held out")
    vb, vs, vy = tb[n_train:].to(dev), ts[n_train:].to(dev), ty[n_train:].to(dev)

    table8 = torch.from_numpy(bucket_table(8)).to(dev)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    nn2 = None
    if os.path.exists("numbav9.py"):
        nn2 = load_nn2("numbav9.py", dev)
        table4 = torch.from_numpy(bucket_table(4)).to(dev)
        l2, m2, p2 = evaluate(lambda iw, ib, s, h: forward(*nn2, 3072, iw, ib, s, h),
                              vb, vs, vy, table4, 4, False)
        print(f"NN2     : held-out loss {l2:.6f}   MAE {m2:.1f} cp   (the number to beat)")
    else:
        print("NN2     : numbav9.py not in this folder, no comparison")

    budget = args.hours * 3600
    t0 = time.time()
    last_log = last_eval = t0
    best = float("inf")
    step = seen = 0
    lr = 0.0
    print(f"run     : {args.hours:.1f} h, batch {args.batch}, lr {args.lr}, K {K:.0f}, "
          f"{NB} buckets x 768 -> {H}, {NHEAD} heads")
    while True:
        frac = (time.time() - t0) / budget
        done = frac >= 1.0 or os.path.exists("STOP")
        if not done:
            lr = args.lr * min(1.0, frac / 0.02) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * frac)))
            lr = max(lr, args.lr * 0.01)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = torch.randint(0, n_train, (args.batch,), device=ddev)
            b, s, y = tb[idx].to(dev, non_blocking=True), ts[idx].to(dev), ty[idx].to(dev).float()
            iw, ib, head = make_idx(b, s, table8, NB, True)
            p = net(iw, ib, s, head) * 400
            loss = ((torch.sigmoid(p / K) - torch.sigmoid(y / K)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            net.clip()
            step += 1
            seen += args.batch
        now = time.time()
        if not done and now - last_log >= 60:
            last_log = now
            print(f"  {(now - t0) / 60:6.1f} min  step {step:>8,}  seen {seen / 1e6:8.1f}M  "
                  f"({seen / (now - t0) / 1e3:,.0f}k pos/s)  lr {lr:.2e}  batch loss {loss.item():.6f}", flush=True)
        if done or now - last_eval >= args.eval_min * 60:
            last_eval = now
            net.eval()
            l3, m3, _ = evaluate(net, vb, vs, vy, table8, NB, True)
            net.train()
            mark = ""
            if l3 < best:
                best = l3
                export(net, "nnue_v3.bin")
                torch.save(net.state_dict(), "nnue_v3.pt")
                mark = "  <- best, saved nnue_v3.bin"
            print(f"  [held-out] {(now - t0) / 60:6.1f} min  loss {l3:.6f}  MAE {m3:.1f} cp{mark}", flush=True)
        if done:
            break

    # ---- final report on the saved (best) network ---------------------------------
    net.load_state_dict(torch.load("nnue_v3.pt", map_location=dev))
    net.eval()
    l3, m3, p3 = evaluate(net, vb, vs, vy, table8, NB, True)
    w1, b1, w2, b2, worst, crc = export(net, "nnue_v3.bin")
    print("\n==================== RESULT ====================")
    print(f"trained        : {(time.time() - t0) / 3600:.2f} h, {seen / 1e6:,.0f}M positions seen")
    print(f"NN3 held-out   : loss {l3:.6f}   MAE {m3:.1f} cp")
    if nn2 is not None:
        print(f"NN2 held-out   : loss {l2:.6f}   MAE {m2:.1f} cp")
        print(f"NN3 vs NN2     : loss {100 * (l2 - l3) / l2:+.1f}% better" if l3 < l2 else
              f"NN3 vs NN2     : NN3 is WORSE by {100 * (l3 - l2) / l2:.1f}%")
        m = p2.abs() < 1000
        a = float((p3[m] * p2[m]).sum() / (p3[m] * p3[m]).sum())
        print(f"engine scale   : NN2 runs at 61/100; NN3 at {round(61 * a)}/100 gives the same cp scale")
    # integer check, exactly as the engine will compute it
    idx = torch.arange(0, VAL_N, VAL_N // 500)
    vbn, vsn = vb[idx.to(dev)].cpu().numpy(), vs[idx.to(dev)].cpu().numpy()
    qi = np.array([eval_int(w1, b1, w2, b2, bucket_table(8), NB, vbn[i], int(vsn[i])) for i in range(len(idx))])
    qf = p3[idx.to(dev)].cpu().numpy()
    print(f"quantisation   : max |int - float| {np.abs(qi - qf).max():.1f} cp over {len(idx)} positions; "
          f"worst-case accumulator {worst} (< 32767)")
    print(f"nnue_v3.bin    : {os.path.getsize('nnue_v3.bin'):,} bytes, crc32 {crc}")
    print("================================================")
    print("DONE -- upload nnue_v3.bin and this output")


if __name__ == "__main__":
    main()