"""Train the evaluation network, quantise it, and export weights for the agent.

    python train.py NNUE

Reads NNUE_feat.npy / NNUE_stm.npy / NNUE_cp.npy from the parser and writes:

    nnue_w1.npy   int16 (770, H)  input weights, row 769 is padding (zeros)
    nnue_b1.npy   int16 (H,)      input bias
    nnue_w2.npy   int16 (H,)      output weights
    nnue_b2.npy   int32 scalar    output bias
    nnue_meta.npy dict            H, QA, QB, and the verification numbers

Architecture: 769 -> H -> 1.

  * 768 inputs are piece-square facts, index = colour*384 + (piece-1)*64 + sq,
    matching the parser and the kernel that will run inside the agent.
  * Input 768 is set when Black is to move. Tempo is worth roughly 20 cp and
    without this the network has to average over both sides.
  * Only about 32 inputs are ever active, so layer one is a sparse lookup and
    add rather than a matrix multiply. That is what makes it fast enough to
    run at every leaf of an alpha-beta search.

The network predicts centipawns directly, but the loss is applied after a
sigmoid. A 30 cp error at level material matters; the same error at +2000
does not, and training on raw centipawns spends most of its effort on
positions that are already decided.

QUANTISATION. The agent has no torch, so the float network has to become
int16 arithmetic that numba can run:

    w1_int = round(w1 * QA)          accumulator is int16
    clipped = clamp(acc, 0, QA)      the activation
    w2_int = round(w2 * QB)
    cp     = dot(clipped, w2_int) // (QA * QB)

Layer-one weights are clipped to +/-1.98 after every optimiser step so the
accumulator cannot overflow int16: 32 features x 1.98 x 127 = 8043, well
inside 32767. The script verifies the quantised network against the float
network at the end and refuses to write the files if they disagree badly --
a quantisation error would show up as an engine that plays slightly wrong
chess for no visible reason, which is the worst kind of bug to ship.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

H = 512                 # hidden width; 1.54x eval cost of the hand-written eval
QA = 127                # input-layer scale; clipped activation lands in 0..127
QB = 64                 # output-layer scale
WEIGHT_CLIP = 1.98      # keeps the int16 accumulator inside range
K = 400.0               # centipawns per unit of the sigmoid
BATCH = 16384
EPOCHS = 60
LR = 3e-3
VAL_FRACTION = 0.01
PAD_INDEX = 769
STM_INDEX = 768
N_INPUTS = 770          # 768 piece features + stm + padding row
OUT_INIT_GAIN = 100.0   # start the output layer in centipawn range


class Net(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.acc = nn.EmbeddingBag(N_INPUTS, hidden, mode="sum",
                                   padding_idx=PAD_INDEX)
        # Bias starts at 0.5, not 0. clamp(x, 0, 1) has zero gradient outside
        # [0, 1], so a unit that begins below zero gets no gradient and can
        # stay dead for the whole run. Starting mid-range keeps every unit
        # alive long enough to learn where it wants to sit.
        self.bias = nn.Parameter(torch.full((hidden,), 0.5))
        self.out = nn.Linear(hidden, 1)
        nn.init.uniform_(self.acc.weight, -0.05, 0.05)
        with torch.no_grad():
            self.acc.weight[PAD_INDEX].zero_()
            # Default initialisation makes the network output about +/-1 cp
            # while targets reach +/-3000, so training wastes its early steps
            # just growing the output scale. Start it in the right range.
            self.out.weight.mul_(OUT_INIT_GAIN)
            self.out.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.acc(features) + self.bias
        hidden = torch.clamp(hidden, 0.0, 1.0)
        return self.out(hidden).squeeze(1)


def load(prefix: str):
    feat = np.load(prefix + "_feat.npy")
    stm = np.load(prefix + "_stm.npy")
    cps = np.load(prefix + "_cp.npy").astype(np.float32)

    n, slots = feat.shape
    print(f"loaded {n:,} positions, {slots} slots")

    # One extra slot for the side-to-move feature. Padding becomes PAD_INDEX,
    # which EmbeddingBag maps to a frozen zero row.
    table = np.full((n, slots + 1), PAD_INDEX, dtype=np.int16)
    table[:, :slots] = np.where(feat < 0, PAD_INDEX, feat)
    table[stm == 1, slots] = STM_INDEX
    return torch.from_numpy(table), torch.from_numpy(cps)


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "NNUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    features, targets = load(prefix)
    n = features.shape[0]

    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(n, generator=generator)
    n_val = max(int(n * VAL_FRACTION), 1)
    val_idx, train_idx = order[:n_val], order[n_val:]
    print(f"train {len(train_idx):,}  validate {len(val_idx):,}\n")

    val_x = features[val_idx].to(device).long()
    val_y = targets[val_idx].to(device)

    model = Net(H).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"network: 769 -> {H} -> 1, {params:,} parameters\n")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
    steps = EPOCHS * (len(train_idx) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=LR, total_steps=steps, pct_start=0.1)

    best_val = float("inf")
    best_state = None
    step = 0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        shuffled = train_idx[torch.randperm(len(train_idx), generator=generator)]
        running = 0.0
        batches = 0
        for start in range(0, len(shuffled) - BATCH + 1, BATCH):
            idx = shuffled[start:start + BATCH]
            batch_x = features[idx].to(device, non_blocking=True).long()
            batch_y = targets[idx].to(device, non_blocking=True)

            predicted = model(batch_x)
            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(predicted / K), torch.sigmoid(batch_y / K))

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step < steps - 1:
                schedule.step()
            step += 1

            # Keep the accumulator inside int16 once quantised.
            with torch.no_grad():
                model.acc.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                model.acc.weight[PAD_INDEX].zero_()

            running += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():
            chunks = [model(val_x[i:i + 65536])
                      for i in range(0, len(val_x), 65536)]
            predicted = torch.cat(chunks)
            val_loss = torch.nn.functional.mse_loss(
                torch.sigmoid(predicted / K), torch.sigmoid(val_y / K)).item()
            val_mae = (predicted - val_y).abs().mean().item()
            # A hidden unit that never leaves the flat part of the clamp is
            # contributing nothing and can never recover. This number should
            # be small; if it is large the initialisation is still wrong.
            activations = (model.acc(val_x[:20000]) + model.bias)
            dead = int(((activations <= 0).all(0) | (activations >= 1).all(0)).sum())

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"epoch {epoch:2d}/{EPOCHS}  train {running/max(batches,1):.5f}  "
              f"val {val_loss:.5f}  val MAE {val_mae:6.1f} cp  "
              f"dead {dead:3d}/{H}  "
              f"{(time.time()-started)/60:5.1f} min{marker}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nbest validation loss {best_val:.5f}")
    torch.save(model.state_dict(), "nnue_float.pt")

    # ---- quantise ---------------------------------------------------------
    w1 = model.acc.weight.detach().cpu().numpy()          # (770, H)
    b1 = model.bias.detach().cpu().numpy()                # (H,)
    w2 = model.out.weight.detach().cpu().numpy()[0]       # (H,)
    b2 = float(model.out.bias.detach().cpu().numpy()[0])

    w1_q = np.round(w1 * QA).astype(np.int32)
    b1_q = np.round(b1 * QA).astype(np.int32)
    w2_q = np.round(w2 * QB).astype(np.int32)
    b2_q = int(round(b2 * QA * QB))

    print("\nquantisation")
    print(f"  w1 int range {w1_q.min():6d} .. {w1_q.max():6d}   (int16 limit 32767)")
    print(f"  w2 int range {w2_q.min():6d} .. {w2_q.max():6d}")
    worst_acc = int(np.abs(b1_q).max() + 33 * np.abs(w1_q).max())
    print(f"  worst-case accumulator {worst_acc:,}   "
          f"{'OK' if worst_acc < 32767 else 'OVERFLOW -- do not ship'}")
    if worst_acc >= 32767 or abs(w1_q).max() > 32767 or abs(w2_q).max() > 32767:
        sys.exit("quantisation out of int16 range -- stop and check this output")

    w1_q = w1_q.astype(np.int16)
    b1_q = b1_q.astype(np.int16)
    w2_q = w2_q.astype(np.int16)

    # ---- verify the integer network against the float network -------------
    def quantised_eval(rows: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            acc = b1_q.astype(np.int32).copy()
            for f in row:
                if f != PAD_INDEX:
                    acc += w1_q[f]
            acc = np.clip(acc, 0, QA)
            out[i] = (int((acc * w2_q.astype(np.int32)).sum()) + b2_q) // (QA * QB)
        return out

    sample = val_x[:2000].cpu().numpy()
    with torch.no_grad():
        float_cp = model(val_x[:2000]).cpu().numpy()
    int_cp = quantised_eval(sample)
    diff = np.abs(int_cp - float_cp)
    print("\nquantisation error vs the float network")
    print(f"  mean {diff.mean():5.2f} cp   median {np.median(diff):5.2f} cp   "
          f"max {diff.max():6.2f} cp")
    if diff.mean() > 5.0:
        sys.exit("quantisation error too large -- stop and check this output")

    np.save("nnue_w1.npy", w1_q)
    np.save("nnue_b1.npy", b1_q)
    np.save("nnue_w2.npy", w2_q)
    np.save("nnue_b2.npy", np.array(b2_q, dtype=np.int32))
    np.save("nnue_meta.npy", np.array(
        {"H": H, "QA": QA, "QB": QB, "val_loss": best_val,
         "quant_mean_err": float(diff.mean())}, dtype=object), allow_pickle=True)

    size = (w1_q.nbytes + b1_q.nbytes + w2_q.nbytes) / 1e6
    print(f"\nwrote nnue_w1/b1/w2/b2.npy -- {size:.2f} MB total")
    print("stop and check this output")


if __name__ == "__main__":
    main()