"""Stream the Lichess Stockfish-evaluation database and stop early.

Pure Python, no WSL, no zstd binary. Works the same on Windows, Mac and Linux.

    pip install zstandard
    python download_evals.py

The file on the server is tens of gigabytes. This decompresses it as it
arrives and closes the connection once TARGET_LINES have been written, so
only the bytes actually needed cross the network.

Safe to interrupt with Ctrl+C: whatever has been written is valid and usable.
Re-running resumes from scratch unless you change OUT_PATH.
"""

import io
import os
import sys
import time
import urllib.request

try:
    import zstandard as zstd
except ImportError:
    sys.exit("zstandard is not installed. Run:  pip install zstandard")

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUT_PATH = "eval_20M.jsonl"
TARGET_LINES = 20_000_000
REPORT_EVERY = 250_000


def main() -> None:
    if os.path.exists(OUT_PATH):
        sys.exit(
            f"{OUT_PATH} already exists. Delete it or change OUT_PATH first, "
            "so a half-finished run is never mistaken for a complete one."
        )

    print(f"source : {URL}")
    print(f"target : {TARGET_LINES:,} lines -> {OUT_PATH}")
    print("no progress bar from the server; this script prints its own\n")

    request = urllib.request.Request(URL, headers={"User-Agent": "chess-nnue-trainer"})
    started = time.time()
    written = 0

    try:
        with urllib.request.urlopen(request) as response:
            reader = zstd.ZstdDecompressor().stream_reader(response)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as out:
                for line in text:
                    out.write(line)
                    written += 1
                    if written % REPORT_EVERY == 0:
                        elapsed = time.time() - started
                        mb = os.fstat(out.fileno()).st_size / 1e6
                        rate = written / elapsed if elapsed else 0.0
                        remaining = (TARGET_LINES - written) / rate if rate else 0.0
                        print(
                            f"  {written:>12,} lines | {mb:8.0f} MB | "
                            f"{rate/1000:6.0f}k lines/s | "
                            f"~{remaining/60:5.1f} min left",
                            flush=True,
                        )
                    if written >= TARGET_LINES:
                        break
    except KeyboardInterrupt:
        print("\ninterrupted -- the lines already written are still valid")
    except Exception as error:
        # A connection reset on the way out is expected: the server is still
        # sending when we stop reading. Anything else is worth seeing.
        print(f"\nstream ended: {error!r}")

    size_mb = os.path.getsize(OUT_PATH) / 1e6 if os.path.exists(OUT_PATH) else 0.0
    print(f"\ndone: {written:,} lines, {size_mb:.0f} MB, "
          f"{(time.time() - started)/60:.1f} min")

    if written:
        with open(OUT_PATH, encoding="utf-8") as check:
            print("\nfirst line:\n" + check.readline().strip()[:400])


if __name__ == "__main__":
    main()