"""Behaviour signature of an engine: static evaluations and fixed-depth searches.

Used two ways:
  * by tests/test_versions.py, in a subprocess, on this engine with some
    compile-time flags switched off;
  * once, on the original single-file engines of earlier versions, to record
    tests/reference/v7_signature.json and v8_signature.json:

        python tests/signature.py path/to/numbav8.py tests/reference/v8_signature.json

Nothing depends on the clock: searches run at fixed depth with the deadline
disabled, from an empty transposition table.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REF = json.loads((Path(__file__).parent / "reference" / "engine_signature.json").read_text())


def signature(A) -> dict:
    out = {"eval": [], "search": []}
    for row in REF["eval"]:
        bb, mb, side, *_ = A.fen_to_state(row["fen"])
        out["eval"].append({"fen": row["fen"], "final": int(A.evaluate_raw(bb, mb, side))})
    for ref in REF["search"]:
        A.U64[:] = 0
        A.I32[:] = 0
        A.I8[:] = 0
        A.I8[:A.TT_SIZE] = -1
        A.I64[:] = 0
        bb, mb, side, castling, ep, half = A.fen_to_state(ref["fen"])
        key = A.compute_key(bb, mb, side, castling, ep)
        A.I64[A.C_ROOT] = side
        A.I64[A.C_DEADLINE] = np.iinfo(np.int64).max
        A.I32[A.EVSTACK_OFF] = A.NO_EVAL
        A.I32[A.EVSTACK_OFF + 1] = A.NO_EVAL
        best, rows = 0, []
        for depth in range(1, len(ref["by_depth"]) + 1):
            mv, sc = A.search_root(bb, mb, side, castling, ep, half, key, depth,
                                   -A.INFINITY, A.INFINITY, best, A.U64, A.I32, A.I8, A.I64)
            best = int(mv)
            rows.append([depth, A.move_to_uci(best), int(sc), int(A.I64[A.C_NODES])])
        out["search"].append({"fen": ref["fen"], "by_depth": rows})
    return out


def engine_with_flags_off(flags: list[str]):
    """Import engine/ with the named compile-time flags set to False."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import engine.config as C
    for f in flags:
        assert hasattr(C, f), f
        setattr(C, f, False)
    from engine import agent
    return agent


if __name__ == "__main__":
    if sys.argv[1] == "--flags-off":                     # test mode: print this engine's signature
        A = engine_with_flags_off(sys.argv[2].split(","))
        print("SIGNATURE " + json.dumps(signature(A)))
    else:                                                # record mode: an original single-file engine
        spec = importlib.util.spec_from_file_location(Path(sys.argv[1]).stem, sys.argv[1])
        A = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = A
        spec.loader.exec_module(A)
        Path(sys.argv[2]).write_text(json.dumps(signature(A), indent=1))
