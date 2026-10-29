"""Training pipeline smoke tests: parse -> train -> export -> check, on a tiny input.

Needs PyTorch and zstandard (requirements-train.txt); skipped without them.
CI runs these in a separate job.

The input is built here, in the Lichess database format, from the 3,071
benchmark positions in training/nn2/bench_v2.npz with their Stockfish scores.
The benchmark has no principal variation, so each line's best move is a stand-in
(the first legal quiet move); it only feeds the parsers' capture filter. This
checks that the pipeline runs and that its output is what the engine loads. It
is not training data, and the networks it makes are meaningless.

Each trainer runs a fixed number of steps with --deterministic, twice, and must
write byte-identical weights both times.
"""
import json
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")
zstandard = pytest.importorskip("zstandard")

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "training" / "nn2" / "bench_v2.npz"
sys.path.insert(0, str(ROOT / "training"))
import regate  # noqa: E402


def lichess_lines() -> bytes:
    d = np.load(BENCH)
    out = []
    for fen, cp in zip(d["fen"], d["sf_cp"]):
        board = chess.Board(str(fen))
        move = next((m for m in board.legal_moves if not board.is_capture(m) and not m.promotion), None)
        if move is not None:
            pv = {"cp": int(cp), "line": move.uci()}       # Lichess scores are from White's side
            out.append(json.dumps({"fen": board.fen(), "evals": [{"pvs": [pv], "knodes": 1000, "depth": 30}]}))
    return ("\n".join(out) + "\n").encode()


def run(args, cwd):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("lichess")
    raw = lichess_lines()
    (d / "sample.jsonl").write_bytes(raw)
    (d / "sample.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(raw))
    return d


def test_third_network_pipeline(sample, tmp_path):
    out = run([ROOT / "training" / "nn3" / "parse_v3.py", "--file", sample / "sample.jsonl.zst", "--workers", 2],
              tmp_path)
    assert "DONE" in out
    assert (tmp_path / "data_v3" / "shard_000.npz").exists()

    train = [ROOT / "training" / "nn3" / "train_v3.py", "--steps", 40, "--eval-steps", 20, "--batch", 256,
             "--val-n", 500, "--seed", 0, "--deterministic"]
    out = run(train, tmp_path)
    assert "DONE" in out
    first = (tmp_path / "nnue_v3.bin").read_bytes()
    run(train, tmp_path)
    assert (tmp_path / "nnue_v3.bin").read_bytes() == first, "same seed and steps must give the same weights"

    # the engine's format: same size as the shipped network, loader's overflow bounds hold,
    # and the gate script's integer evaluator (identical to the engine's kernel) reads it
    assert len(first) == (ROOT / "engine" / "weights" / "nnue_v3.bin").stat().st_size
    net = regate.Net("smoke", tmp_path / "nnue_v3.bin", 8, 384, regate.bucket8)
    assert int(np.abs(net.b1).max()) + 32 * int(np.abs(net.w1).max()) < 32767
    assert int(np.abs(net.w2).max()) < 8256
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert net.eval_stm(fen) == net.eval_stm(chess.Board(fen).mirror().fen())


def test_second_network_pipeline(sample, tmp_path):
    out = run([ROOT / "training" / "nn2" / "parse_v2.py", "--input", sample / "sample.jsonl", "--data-dir", "data"],
              tmp_path)
    assert "DONE" in out
    train = [ROOT / "training" / "nn2" / "train_v2.py", "--data-dir", "data", "--steps", 40, "--batch", 256,
             "--deterministic", "--bench", BENCH]
    run(train, tmp_path)
    first = (tmp_path / "nnue_v2.bin").read_bytes()
    (tmp_path / "data" / "nnue_v2.ckpt").unlink()
    run(train, tmp_path)
    assert (tmp_path / "nnue_v2.bin").read_bytes() == first, "same seed and steps must give the same weights"

    # the gate reads the trainer's float file and runs every check (this network fails them)
    (tmp_path / "bench_v2.npz").write_bytes(BENCH.read_bytes())
    out = run([ROOT / "training" / "nn2" / "gate_v2.py", "nnue_v2.bin"], tmp_path)
    assert "VERDICT" in out and out.count("[PASS]") + out.count("[FAIL]") == 7
