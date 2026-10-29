"""Engine tests. Run from the repo root:  python -m pytest tests -v

Importing the engine compiles every kernel, so the first test takes about a
minute. Nothing here depends on the clock: searches run at fixed depth with
the deadline disabled, so results are exact and repeatable.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest
from numba import njit

from engine import agent as E

REF = json.loads((Path(__file__).parent / "reference" / "engine_signature.json").read_text())

# Published perft node counts (chessprogramming.org "Perft Results"), depth 1..n.
PERFT = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [20, 400, 8902, 197281, 4865609]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862, 4085603]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467, 422333]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379, 2103487]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890, 3894594]),
]

_gen, _make, _unmake, _MAX = E.gen_legal, E.make_move, E.unmake_move, E.MAX_MOVES


@njit(cache=False)
def _perft(bb, mb, side, castling, ep, depth, ply, moves, undo):
    base = ply * _MAX
    n = _gen(bb, mb, side, castling, ep, moves, base, False)
    if depth == 1:
        return n - base
    total = 0
    for i in range(base, n):
        m = moves[i]
        c2, e2, h2, k2 = _make(bb, mb, side, castling, ep, 0, np.uint64(0), m, undo, ply)
        total += _perft(bb, mb, 1 - side, c2, e2, depth - 1, ply + 1, moves, undo)
        _unmake(bb, mb, side, m, undo, ply)
    return total


def test_weights_loaded():
    """Both networks passed their checksum, size and overflow checks at import."""
    assert int(E.NNUE_OK[0]) == 1
    assert int(E.NN2_OK[0]) == 1


@pytest.mark.parametrize("fen,expected", PERFT)
def test_perft(fen, expected):
    """Legal move generation and make/unmake reproduce published node counts."""
    moves = np.zeros(16 * _MAX, dtype=np.int32)
    undo = np.zeros(16, dtype=np.int8)
    bb, mb, side, castling, ep, _ = E.fen_to_state(fen)
    got = [int(_perft(bb, mb, side, castling, ep, d, 0, moves, undo)) for d in range(1, len(expected) + 1)]
    assert got == expected


@pytest.mark.parametrize("row", REF["eval"], ids=lambda r: r["fen"].split()[0][:20])
def test_evaluators_match_reference(row):
    """Each evaluator returns what the original single-file engine returned."""
    bb, mb, side, *_ = E.fen_to_state(row["fen"])
    assert int(E.evaluate_raw(bb, mb, side)) == row["final"]
    assert int(E.nnue2_raw(bb, mb, side)) == row["nn2_raw"]
    assert int(E.nnue_raw(bb, mb, side)) == row["nn1_raw"]


def _reset():
    E.U64[:] = 0
    E.I32[:] = 0
    E.I8[:] = 0
    E.I8[:E.TT_SIZE] = -1
    E.I64[:] = 0


@pytest.mark.parametrize("ref", REF["search"], ids=lambda r: r["fen"].split()[0][:20])
def test_search_matches_reference(ref):
    """Fixed-depth search gives the same move, score and node count at every
    depth as the original single-file engine."""
    _reset()
    bb, mb, side, castling, ep, half = E.fen_to_state(ref["fen"])
    key = E.compute_key(bb, mb, side, castling, ep)
    assert str(int(key)) == ref["key"]
    E.I64[E.C_ROOT] = side
    E.I64[E.C_DEADLINE] = np.iinfo(np.int64).max
    E.I32[E.EVSTACK_OFF] = E.NO_EVAL
    E.I32[E.EVSTACK_OFF + 1] = E.NO_EVAL
    best = 0
    for depth, uci, score, nodes in ref["by_depth"]:
        mv, sc = E.search_root(bb, mb, side, castling, ep, half, key, depth,
                               -E.INFINITY, E.INFINITY, best, E.U64, E.I32, E.I8, E.I64)
        best = int(mv)
        assert (E.move_to_uci(best), int(sc), int(E.I64[E.C_NODES])) == (uci, score, nodes)


@pytest.mark.parametrize("row", REF["eval"], ids=lambda r: r["fen"].split()[0][:20])
def test_main_network_colour_symmetry(row):
    """Swapping colours and flipping the board must not change the main
    (third) network's score for the side to move."""
    bb, mb, side, *_ = E.fen_to_state(row["fen"])
    bb2, mb2, side2, *_ = E.fen_to_state(chess.Board(row["fen"]).mirror().fen())
    assert int(E.nnue2_raw(bb, mb, side)) == int(E.nnue2_raw(bb2, mb2, side2))


# ---- training/regate.py's numpy evaluator is the engine's kernel -------------------------
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("regate", Path(__file__).parents[1] / "training" / "regate.py")
REGATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REGATE)


def test_regate_matches_engine_third_network():
    """The gate script scores the engine's network exactly as the engine does."""
    net = REGATE.THIRD()
    for row in REF["eval"]:
        assert net.eval_stm(row["fen"]) == row["nn2_raw"]


def test_regate_matches_version_9_second_network():
    """training/nn2/nnue_v2_int16.bin, scored by the gate script, reproduces what the
    original version 9 engine returned for the second network."""
    net = REGATE.SECOND()
    for row in REF["second_network_v9"]:
        assert net.eval_stm(row["fen"]) == row["nn2_raw"]
