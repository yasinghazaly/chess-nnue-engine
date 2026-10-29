"""Start positions for the matches.

The harness ships 8 curated positions (harness/rules.py), which is only 16 games
before every pairing repeats. These 40 extra lines are standard main lines of
common openings, 6 to 10 plies deep, so the start positions are balanced by
established theory rather than by either engine under test. Each position is
played twice in a match, once with each engine as White.
"""
from __future__ import annotations

import chess

from harness.rules import OPENINGS as HARNESS_OPENINGS

LINES = [
    ("Ruy Lopez, Closed", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"),
    ("Ruy Lopez, Berlin", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6"),
    ("Italian, Giuoco Piano", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6"),
    ("Two Knights", "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O"),
    ("Scotch", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6"),
    ("Four Knights", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O"),
    ("Petroff", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5"),
    ("Vienna", "e4 e5 Nc3 Nf6 Bc4 Nc6 d3 Bc5 Nf3 d6"),
    ("Sicilian, Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("Sicilian, Classical", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6"),
    ("Sicilian, Taimanov", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7"),
    ("Sicilian, Kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6"),
    ("Sicilian, Alapin", "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6"),
    ("Sicilian, Rossolimo", "e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Re1 Nf6"),
    ("French, Classical", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7"),
    ("French, Tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5"),
    ("French, Advance", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6"),
    ("Caro-Kann, Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6"),
    ("Caro-Kann, Advance", "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5"),
    ("Scandinavian", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6"),
    ("Pirc", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O"),
    ("Alekhine, Modern", "e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O"),
    ("Queen's Gambit Accepted", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5"),
    ("Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5"),
    ("Semi-Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7"),
    ("Nimzo-Indian", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5"),
    ("Queen's Indian", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7"),
    ("King's Indian", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O"),
    ("Grunfeld", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3"),
    ("Catalan", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O"),
    ("Benoni", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6"),
    ("Dutch, Stonewall", "d4 f5 g3 Nf6 Bg2 e6 Nf3 d5 O-O Bd6"),
    ("London System", "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 e6"),
    ("Trompowsky", "d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nf3 d6"),
    ("English, Symmetrical", "c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 Nf6"),
    ("English, Reversed Sicilian", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5"),
    ("Reti", "Nf3 d5 g3 Nf6 Bg2 c6 O-O Bg4 d3 Nbd7"),
    ("King's Indian Attack", "Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d3 d6"),
    ("Bird", "f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O"),
]


def _fen(moves: str) -> str:
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    return board.fen()


OPENINGS: list[tuple[str, str]] = list(HARNESS_OPENINGS) + [(name, _fen(m)) for name, m in LINES]
