"""Version 10 differs from versions 7 and 8 only in the evaluation.

With its network flag off, this engine must behave exactly like the original
version 8 file; with both network flags off, exactly like version 7: the same
static evaluations, and the same move, score and node count at every depth of a
fixed-depth search. The references were recorded from the original single-file
engines with tests/signature.py.

So move generation and search are unchanged since version 7, and the match
results in the README compare evaluations.

Each case compiles the engine again in a subprocess (flags are compile-time),
which takes about a minute.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "tests" / "reference"


@pytest.mark.parametrize("version,flags", [("v8", "USE_NNUE_V2"), ("v7", "USE_NNUE_V2,USE_NNUE")])
def test_flags_off_equals_earlier_version(version, flags):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "signature.py"), "--flags-off", flags],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(x for x in r.stdout.splitlines() if x.startswith("SIGNATURE "))
    got = json.loads(line[len("SIGNATURE "):])
    want = json.loads((REFS / f"{version}_signature.json").read_text())
    assert got["eval"] == want["eval"]
    for g, w in zip(got["search"], want["search"], strict=True):
        assert g == w, w["fen"]
