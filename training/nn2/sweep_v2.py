"""Sweep v2 -- find out WHY the 60-minute run degraded, fix it, and keep the best.

    python sweep_v2.py                          # 4 x 8 min tests, then a 20 min final run
    python sweep_v2.py --minutes 6 --final-minutes 15
    python sweep_v2.py --final-minutes 0        # tests only
    python sweep_v2.py --steps-per-second 40    # fixed schedule: same steps on any hardware

Needs train_v2.py, bench_v2.npz and select_v2.npz in the same folder.
Stop early at any time with:  touch STOP   (the best network so far is kept)

What the 60-minute run showed
-----------------------------
Held-out loss improved at every checkpoint, but the error in centipawns got
worse (198 -> 205) and the benchmark peaked after two minutes, then slid below
v7. The network was buying probability-space accuracy by pushing large
evaluations to extremes. Two candidate causes, tested in a 2 x 2 design so each
one's effect is measured on its own rather than guessed:

  1. MATING-ATTACK LABELS. Of the 34.8 M labels sitting at the +-2000 clamp
     (31 M of them mates), many are forced mates with roughly equal material.
     No static evaluator can see a forced mate; trying to fit one teaches the
     network to scream +2000 at any exposed king. The filter drops clamped
     labels unless the winning side is also ahead by >= 3 pawns of material.
     Won endgames, the thing we kept the mates for, survive the filter.

  2. LOSS SATURATION. sigmoid(eval / 400) is nearly flat beyond +-800 cp, so
     the network is barely penalised for calling a +700 position +1800.
     K = 800 keeps the loss sensitive across the whole range the engine uses.

Runs: A = control (the 60-minute settings), B = filter, C = K 800,
D = filter + K 800. Every run is judged on select_v2.npz: positions from
v7 self-play games played for this purpose, Stockfish-labelled, never used by the
gate. The gate's own benchmark is printed for information only and is not used
to choose anything, so gate_v2.py remains an honest test afterwards.

The winning settings then get one longer run, and whichever checkpoint scores
best on the selection set (from any run) is written to nnue_v2.bin.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import shutil
import threading
import time

import numpy as np
import torch

import train_v2 as T

VAL = np.array([1, 3, 3, 5, 9, 0], dtype=np.int16)
CLAMP_LEVEL = 1900      # labels at or beyond this are "at the clamp" (mates and clipped scores)
MIN_EDGE = 3            # keep a clamped label only if the winner is up >= 3 pawns of material


def material_edge(feat, chunk=4_000_000):
    """White-minus-Black material per row, computed in chunks to bound memory."""
    out = np.empty(len(feat), dtype=np.int16)
    for i in range(0, len(feat), chunk):
        f = feat[i:i + chunk].astype(np.int32)
        valid = f >= 0
        f0 = np.where(valid, f, 0)
        col = f0 // 384
        pt = (f0 % 384) // 64
        out[i:i + chunk] = np.where(valid, VAL[pt] * (1 - 2 * col), 0).sum(1)
    return out


class Sampler:
    """Random batches from an optional index pool, gathered in a background thread."""

    def __init__(self, f, s, c, pool, bsz, device):
        self.f, self.s, self.c, self.pool = f, s, c, pool
        self.n = len(pool) if pool is not None else len(c)
        self.bsz, self.device = bsz, device
        self.q = queue.Queue(maxsize=6)
        self.alive = True
        self.th = threading.Thread(target=self._fill, daemon=True)
        self.th.start()

    def _fill(self):
        pin = self.device.type == "cuda"
        while self.alive:
            r = torch.randint(self.n, (self.bsz,))
            i = self.pool.index_select(0, r).long() if self.pool is not None else r
            b = (self.f.index_select(0, i), self.s.index_select(0, i), self.c.index_select(0, i))
            if pin:
                b = tuple(t.pin_memory() for t in b)
            self.q.put(b)

    def next(self):
        f, s, c = self.q.get()
        return (f.to(self.device, non_blocking=True), s.to(self.device, non_blocking=True),
                c.to(self.device, non_blocking=True))

    def close(self):
        self.alive = False
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


class EvalSet(T.Bench):
    """Same scoring as the gate's benchmark, on any npz with fen / sf_cp / quiet / strict."""

    @torch.no_grad()
    def full(self, net):
        net.eval()
        y = net(self.feat, self.stm).float().cpu().numpy() * T.EVAL_SCALE * self.sign
        net.train()
        q = float(np.corrcoef(y[self.quiet], self.sf[self.quiet])[0, 1])
        s = float(np.corrcoef(y[self.strict], self.sf[self.strict])[0, 1])
        return q, s, spearman(y[self.quiet], self.sf[self.quiet])


def run(name, K, pool, minutes, data, dev, sel, gate, vset, args, stop_file, rate_hint):
    f_t, s_t, c_t = data
    torch.manual_seed(args.seed)
    net = T.Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    samp = Sampler(f_t, s_t, c_t, pool, args.batch, dev)

    def loss_fn(y, s, c):
        tgt = c.float() * (1.0 - 2.0 * s.float())
        return (torch.sigmoid(y * (T.EVAL_SCALE / K)) - torch.sigmoid(tgt / K)).square().mean()

    total = max(400, int(rate_hint * minutes * 60)) if rate_hint else None
    step = 0

    def lr_at(st):
        tot = total or 10 ** 9
        warm = min(1.0, (st + 1) / 200)
        prog = min(1.0, st / tot)
        return args.lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))

    # speed measurement on the first run only
    t0 = time.time()
    if total is None:
        while time.time() - t0 < 20:
            fb, sb, cb = samp.next()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = loss_fn(net(fb, sb), sb, cb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            net.clip_()
            step += 1
        if dev.type == "cuda":
            torch.cuda.synchronize()
        rate_hint = step / (time.time() - t0)
        total = max(400, int(rate_hint * minutes * 60))
        print(f"  speed {rate_hint * args.batch / 1e6:.2f} M positions/s -> {total:,} steps per run", flush=True)

    n_ck = 8
    ck_at = set(int(total * (k + 1) / n_ck) for k in range(n_ck))
    best = (-9.0, None, None)
    stopped = False
    print(f"  {'min':>5} {'step':>8} | {'select q':>8} {'strict':>7} {'rank':>6} | {'gate q':>7} {'strict':>7} | "
          f"{'val MAE':>7}", flush=True)
    while step < total:
        fb, sb, cb = samp.next()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = loss_fn(net(fb, sb), sb, cb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        net.clip_()
        step += 1
        if step in ck_at:
            sq, ss, sr = sel.full(net)
            gq, gs, _ = gate.full(net)
            vmae = val_mae(net, vset, dev)
            score = 0.5 * (sq + ss)
            mark = ""
            if score > best[0]:
                path = f"sweep_{name}.bin"
                net.export(path, {"run": name, "K": K, "filter": pool is not None, "step": step,
                                  "select_quiet": round(sq, 4), "select_strict": round(ss, 4)})
                best = (score, step, (sq, ss, sr, gq, gs, vmae))
                mark = "  *best"
            print(f"  {(time.time() - t0) / 60:5.1f} {step:8,} | {sq:8.3f} {ss:7.3f} {sr:6.3f} | {gq:7.3f} {gs:7.3f} | "
                  f"{vmae:7.1f}{mark}", flush=True)
        if step % 100 == 0 and os.path.exists(stop_file):
            stopped = True
            break
    samp.close()
    return best, rate_hint, stopped


@torch.no_grad()
def val_mae(net, vset, dev):
    vf, vs, vc = vset
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(vc), 32768):
        f, s, c = vf[i:i + 32768].to(dev), vs[i:i + 32768].to(dev), vc[i:i + 32768].to(dev)
        y = net(f, s) * T.EVAL_SCALE
        tot += (y - c.float() * (1 - 2 * s.float())).abs().sum().item()
        n += len(c)
    net.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--minutes", type=float, default=8.0, help="length of each of the four test runs")
    ap.add_argument("--final-minutes", type=float, default=20.0, help="longer run with the winning settings")
    ap.add_argument("--steps-per-second", type=float, default=0.0,
                    help="fix the schedule (steps = this x minutes x 60) instead of measuring speed; "
                         "the default, 0, measures it as the original runs did")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--only", default="", help="e.g. --only BD to run just those tests")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    stop_file = os.path.abspath("STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    print(f"device   : {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}", flush=True)

    t0 = time.time()
    feat, stm, cp = T.load_data(args.data_dir, args.max_rows)
    n = len(cp)
    n_val = min(200_000, n // 20)
    n_tr = n - n_val
    print(f"data     : {n:,} positions in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    edge = material_edge(feat[:n_tr])
    at_clamp = np.abs(cp[:n_tr].astype(np.int32)) >= CLAMP_LEVEL
    winner_edge = edge.astype(np.int32) * np.sign(cp[:n_tr].astype(np.int32))
    drop = at_clamp & (winner_edge < MIN_EDGE)
    pool_np = np.flatnonzero(~drop).astype(np.int32)
    print(f"filter   : {at_clamp.sum():,} labels at the clamp; {drop.sum():,} of them dropped "
          f"(winner not >= {MIN_EDGE} pawns up) = {100 * drop.sum() / n_tr:.1f}% of the data  "
          f"[{time.time() - t1:.0f}s]", flush=True)
    del edge, at_clamp, winner_edge, drop

    data = (torch.from_numpy(feat[:n_tr]), torch.from_numpy(stm[:n_tr]), torch.from_numpy(cp[:n_tr]))
    vset = (torch.from_numpy(feat[n_tr:]), torch.from_numpy(stm[n_tr:]), torch.from_numpy(cp[n_tr:]))
    pool = torch.from_numpy(pool_np)
    sel = EvalSet("select_v2.npz", dev)
    gate = EvalSet("bench_v2.npz", dev)
    print(f"select   : {sel.quiet.sum()} quiet / {sel.strict.sum()} strict positions from v7 self-play games "
          f"(chooses checkpoints)", flush=True)
    print(f"gate     : shown for information only, never used to choose\n", flush=True)

    configs = [("A", 400.0, False, "control: the 60-minute settings"),
               ("B", 400.0, True, "drop mating-attack labels"),
               ("C", 800.0, False, "wider loss, K = 800"),
               ("D", 800.0, True, "both fixes")]
    if args.only:
        configs = [c for c in configs if c[0] in args.only.upper()]

    results = {}
    rate = args.steps_per_second or None     # None: measured on the first run, as run originally
    for name, K, filt, desc in configs:
        print(f"run {name}: {desc}  ({args.minutes:.0f} min)", flush=True)
        best, rate, stopped = run(name, K, pool if filt else None, args.minutes, data, dev, sel, gate, vset,
                                  args, stop_file, rate)
        results[name] = (best, K, filt, desc)
        print(flush=True)
        if stopped:
            break

    print("=" * 86)
    print("SUMMARY  (best checkpoint of each run, chosen on the selection set)")
    print(f"  {'run':4s} {'settings':34s} | {'select q':>8} {'strict':>7} {'rank':>6} | {'gate q':>7} {'strict':>7}")
    for name, (best, K, filt, desc) in results.items():
        if best[2] is None:
            continue
        sq, ss, sr, gq, gs, vm = best[2]
        print(f"  {name:4s} {desc:34s} | {sq:8.3f} {ss:7.3f} {sr:6.3f} | {gq:7.3f} {gs:7.3f}")
    print(f"  {'':4s} {'v7 hand-written, for reference':34s} |    0.902   0.906        |   0.868   0.905")
    print(f"  {'':4s} {'v8 network, for reference':34s} |    0.890   0.889        |   0.850   0.843")
    done = [(v[0][0], k) for k, v in results.items() if v[0][2] is not None]
    if not done:
        print("no completed checkpoints")
        return
    winner = max(done)[1]
    wbest, wK, wfilt, wdesc = results[winner]
    print(f"\n  winner: run {winner} ({wdesc})")
    if "A" in results and results["A"][0][2] is not None:
        a = results["A"][0][0]
        for nm in ("B", "C", "D"):
            if nm in results and results[nm][0][2] is not None:
                print(f"  effect vs control, run {nm}: {results[nm][0][0] - a:+.3f} selection score")

    overall = (wbest[0], f"sweep_{winner}.bin")
    if args.final_minutes > 0 and not os.path.exists(stop_file):
        print(f"\nfinal run with the winning settings ({args.final_minutes:.0f} min)", flush=True)
        fbest, _, _ = run("final", wK, pool if wfilt else None, args.final_minutes, data, dev, sel, gate, vset,
                          args, stop_file, rate)
        if fbest[2] is not None and fbest[0] > overall[0]:
            overall = (fbest[0], "sweep_final.bin")
    shutil.copyfile(overall[1], "nnue_v2.bin")
    print(f"\nbest network: {overall[1]} (selection score {overall[0]:.3f}) -> copied to nnue_v2.bin")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    print("next     : python gate_v2.py")
    print("DONE")


if __name__ == "__main__":
    main()
