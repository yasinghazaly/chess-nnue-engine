"""The single-file build must behave exactly like the engine package.

Builds agent.py into a temporary folder, imports it, and checks evaluations
and fixed-depth search results against the same reference as test_engine.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = json.loads((Path(__file__).parent / "reference" / "engine_signature.json").read_text())


def test_single_file_build_matches_reference(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_submission
    finally:
        sys.path.pop(0)
    out = tmp_path / "agent.py"
    out.write_text(build_submission.build(), encoding="utf-8", newline="\n")

    spec = importlib.util.spec_from_file_location("built_agent", out)
    B = importlib.util.module_from_spec(spec)
    # Numba resolves compiled functions through sys.modules, so the module
    # has to be registered before it runs.
    sys.modules["built_agent"] = B
    spec.loader.exec_module(B)

    assert int(B.NNUE_OK[0]) == 1 and int(B.NN2_OK[0]) == 1

    for row in REF["eval"]:
        bb, mb, side, *_ = B.fen_to_state(row["fen"])
        assert int(B.evaluate_raw(bb, mb, side)) == row["final"], row["fen"]

    for ref in REF["search"]:
        B.U64[:] = 0
        B.I32[:] = 0
        B.I8[:] = 0
        B.I8[:B.TT_SIZE] = -1
        B.I64[:] = 0
        bb, mb, side, castling, ep, half = B.fen_to_state(ref["fen"])
        key = B.compute_key(bb, mb, side, castling, ep)
        B.I64[B.C_ROOT] = side
        B.I64[B.C_DEADLINE] = np.iinfo(np.int64).max
        B.I32[B.EVSTACK_OFF] = B.NO_EVAL
        B.I32[B.EVSTACK_OFF + 1] = B.NO_EVAL
        best = 0
        for depth, uci, score, nodes in ref["by_depth"]:
            mv, sc = B.search_root(bb, mb, side, castling, ep, half, key, depth,
                                   -B.INFINITY, B.INFINITY, best, B.U64, B.I32, B.I8, B.I64)
            best = int(mv)
            assert (B.move_to_uci(best), int(sc), int(B.I64[B.C_NODES])) == (uci, score, nodes), ref["fen"]
