"""Filter tests for the second- and third-network parsers.

Each case is one hand-made line in the Lichess evaluation format, built to hit
exactly one rule. The tests call each parser's worker function directly
(`parse_chunk`), so nothing is downloaded and zstandard is not needed.

Expected features and boards are computed independently with python-chess, not
with the parsers' own board code.
"""
import importlib.util
import json
from pathlib import Path

import chess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P2 = _load("parse_v2", "training/nn2/parse_v2.py")
P3 = _load("parse_v3", "training/nn3/parse_v3.py")


def line(fen, move, cp=None, mate=None, depth=30, evals=None):
    pv = {"line": f"{move} e7e6"}
    if mate is not None:
        pv["mate"] = mate
    else:
        pv["cp"] = cp
    ev = evals if evals is not None else [{"pvs": [pv], "knodes": 1000, "depth": depth}]
    return json.dumps({"fen": fen, "evals": ev}).encode()


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
CAPTURE = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"      # e4xd5
EN_PASSANT = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"  # e5xf6 e.p.
PROMOTION = "8/P6k/8/8/8/8/8/4K3 w - - 0 1"                                  # a7-a8=Q
IN_CHECK = "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"                                  # rook a1 checks e1
MATE_UP = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"                             # Ra8#, White a rook up
MATE_LEVEL = "6k1/5ppp/8/8/8/8/r4PPP/1R4K1 w - - 0 1"                        # Rb8#, material level
QUEEN_UP = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"

CASES = {
    #  name           (line,                                                    v2 outcome,  v3 outcome)
    "quiet":         (line(START, "e2e4", cp=30),                               "kept",      "kept"),
    "black_to_move": (line(AFTER_E4, "e7e5", cp=40),                            "kept",      "kept"),
    "capture":       (line(CAPTURE, "e4d5", cp=20),                             "capture",   "capture"),
    "en_passant":    (line(EN_PASSANT, "e5f6", cp=60),                          "capture",   "capture"),
    "promotion":     (line(PROMOTION, "a7a8q", cp=900),                         "promotion", "promotion"),
    "in_check":      (line(IN_CHECK, "e1e2", cp=-500),                          "in_check",  "in_check"),
    "shallow":       (line(START, "e2e4", cp=30, depth=10),                     "shallow",   "kept"),
    "mate_up":       (line(MATE_UP, "a1a8", mate=1),                            "kept",      "kept"),
    "mate_level":    (line(MATE_LEVEL, "b1b8", mate=1),                         "kept",      "clamp_level"),
    "big_score":     (line(QUEEN_UP, "a1a2", cp=3500),                          "kept",      "kept"),
    "malformed":     (b"this is not json",                                      "malformed", "malformed"),
    "deepest_wins":  (line(START, "e2e4", evals=[
                         {"pvs": [{"cp": 10, "line": "e2e4"}], "depth": 20, "knodes": 10},
                         {"pvs": [{"cp": 50, "line": "d2d4"}], "depth": 34, "knodes": 10}]),
                      "kept", "kept"),
}

V2_REASON = {"kept": P2.R_KEPT, "capture": P2.R_CAPTURE, "promotion": P2.R_PROMO,
             "in_check": P2.R_CHECK, "shallow": P2.R_SHALLOW, "malformed": P2.R_MALFORMED}
V3_REASON = {"kept": "kept", "capture": "capture", "promotion": "promotion",
             "in_check": "in_check", "malformed": "malformed", "clamp_level": "clamp_level_material"}


def v2_features(fen):
    """Expected parse_v2 features: colour*384 + (type-1)*64 + square, sorted, -1 padded."""
    b = chess.Board(fen)
    idx = sorted((0 if p.color else 1) * 384 + (p.piece_type - 1) * 64 + sq for sq, p in b.piece_map().items())
    return idx + [-1] * (32 - len(idx))


def v3_codes(fen):
    """Expected parse_v3 board codes: 1-6 white P N B R Q K, 7-12 black, 0 empty."""
    b = chess.Board(fen)
    out = [0] * 64
    for sq, p in b.piece_map().items():
        out[sq] = p.piece_type + (0 if p.color else 6)
    return out


# ---- one case at a time: each line lands in the expected bucket -------------------------
@pytest.mark.parametrize("name", CASES)
def test_parse_v2_filter(name):
    raw, outcome, _ = CASES[name]
    feat, stm, cp, tally = P2.parse_chunk([raw], P2.MIN_DEPTH)
    expected = np.zeros(P2.N_REASONS, dtype=np.int64)
    expected[V2_REASON[outcome]] = 1
    got = tally.copy()
    got[P2.R_MATE_KEPT] = got[P2.R_CLIPPED] = 0          # informational tallies, checked below
    assert got.tolist() == expected.tolist()
    assert len(cp) == (1 if outcome == "kept" else 0)


@pytest.mark.parametrize("name", CASES)
def test_parse_v3_filter(name):
    raw, _, outcome = CASES[name]
    _, _, score, cnt = P3.parse_chunk(raw + b"\n")
    assert cnt.get(V3_REASON[outcome]) == 1
    assert cnt["lines"] == 1
    assert len(score) == (1 if outcome == "kept" else 0)


# ---- what the kept rows contain ----------------------------------------------------------
def test_parse_v2_kept_rows():
    names = ["quiet", "black_to_move", "mate_up", "big_score", "deepest_wins"]
    feat, stm, cp, tally = P2.parse_chunk([CASES[n][0] for n in names], P2.MIN_DEPTH)
    fens = [START, AFTER_E4, MATE_UP, QUEEN_UP, START]
    assert feat.dtype == np.int16 and feat.shape == (5, 32)
    assert [list(r) for r in feat] == [v2_features(f) for f in fens]
    assert stm.tolist() == [0, 1, 0, 0, 0]
    # White's point of view throughout; mate and big score clamped to +2000; deepest eval used
    assert cp.tolist() == [30, 40, P2.CP_CLIP, P2.CP_CLIP, 50]
    assert tally[P2.R_MATE_KEPT] == 1 and tally[P2.R_CLIPPED] == 1


def test_parse_v3_kept_rows():
    names = ["quiet", "black_to_move", "mate_up", "big_score", "deepest_wins"]
    packed, stm, score, cnt = P3.parse_chunk(b"\n".join(CASES[n][0] for n in names) + b"\n")
    fens = [START, AFTER_E4, MATE_UP, QUEEN_UP, START]
    assert packed.dtype == np.uint8 and packed.shape == (5, 32)
    unpacked = np.empty((5, 64), dtype=np.uint8)
    unpacked[:, 0::2] = packed & 0x0F                     # low nibble = even square
    unpacked[:, 1::2] = packed >> 4
    assert [list(r) for r in unpacked] == [v3_codes(f) for f in fens]
    assert stm.tolist() == [0, 1, 0, 0, 0]
    # side to move's point of view: Black's +40 for White is stored as -40
    assert score.tolist() == [30, -40, P3.CLAMP, P3.CLAMP, 50]
    assert cnt["kept"] == 5


def test_parse_v3_clamp_rule_is_symmetric():
    """A clamped label survives when the winning side is >= 3 pawns up, for either colour."""
    black_up = "4k3/8/8/8/8/q7/8/4K3 b - - 0 1"           # Black a queen up, Black to move
    level = "4k3/q7/8/8/8/8/8/Q3K3 b - - 0 1"             # queens each
    _, _, score, cnt = P3.parse_chunk(line(black_up, "a3a4", cp=-3000) + b"\n"
                                      + line(level, "a7a6", cp=-3000) + b"\n")
    assert cnt["kept"] == 1 and cnt["clamp_level_material"] == 1
    assert score.tolist() == [P3.CLAMP]                   # -2000 for White = +2000 for Black to move
