"""Head-to-head matches between engine versions, in the starter harness, with a
sequential test that stops each match as soon as the result is statistically clear.

    python match.py                                   # all matches, default settings
    python match.py --matches v9:v7 --workers 8       # one match
    python match.py --report                          # print results so far, play nothing
    touch STOP                                        # finish the games in flight, then end the run

Run from the root of a clone of the starter harness (github.com/advitrocks9/
aichessathon-starter), with this file, openings.py and engines/<name>/agent.py
next to its harness/ folder.

Each match plays pairs of games: every start position once with each engine as
White. Games run through harness.referee.play_match and harness.sandbox.local,
unchanged, so clock, legality, adjudication and process handling are the
platform's. Several games run at once (--workers); each engine still gets one
process per game and is suspended while its opponent thinks, as on the platform.

Stopping rule (fixed before any game is played)
-----------------------------------------------
A generalised sequential probability ratio test (GSPRT) on game pairs
("pentanomial": a pair scores 0, 0.5, 1, 1.5 or 2 points), the method used to
test changes to open-source chess engines. For a match A:B with bounds
[elo0, elo1] it weighs

    H0: A is elo0 stronger than B    against    H1: A is elo1 stronger than B

and, once at least --min-games (default 40) games are in, stops when the
log-likelihood ratio leaves (ln(beta/(1-alpha)), ln((1-beta)/alpha)),
i.e. +-2.94 for alpha = beta = 0.05. Crossing the upper bound accepts H1, the lower
accepts H0. If neither happens by --max-games the match stops there and is reported
as inconclusive, with its score and 95% interval. Results are written whatever they
show.

Output, per match, in results/<A>_vs_<B>/:
    games.jsonl    one line per game (resumable: finished games are not replayed)
    pgn/           every game
    summary.json   counts, score, interval, Elo, LLR, verdict, engine hashes, settings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.sandbox import local  # noqa: E402
from openings import OPENINGS  # noqa: E402

# A:B:elo0:elo1. Every match asks the same question of A against B: is A at least
# +35 Elo stronger (the old 55 % ship rule), or not stronger at all (+0)?
# v9:v10 is put that way round because the claim being tested is "v9 is better".
DEFAULT_MATCHES = "v9:v7:0:35,v8:v7:0:35,v10:v7:0:35,v9:v8:0:35,v9:v10:0:35"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(s / (1.0 - s))


def gsprt_llr(values: list[float], elo0: float, elo1: float) -> float:
    """Normal-approximation GSPRT on per-observation scores in [0, 1].

    LLR = n / (2 var) * ((mean - s0)^2 - (mean - s1)^2), var the sample variance.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 0.0:
        var = 1e-3
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n / (2.0 * var) * ((mean - s0) ** 2 - (mean - s1) ** 2)


def interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% interval of per-observation scores (same formula as harness/arena.py)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    spread = sum((v - mean) ** 2 for v in values)
    margin = 1.96 * math.sqrt(spread / (n - 1) / n)
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


# --------------------------------------------------------------------------
# One match
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(index: int) -> tuple[int, bool]:
    """Game index -> (opening index, A plays White). Pairs share an opening."""
    return (index // 2) % len(OPENINGS), index % 2 == 0


def play_one(index: int, a: Path, b: Path, args: argparse.Namespace) -> dict:
    op, a_white = schedule(index)
    name, fen = OPENINGS[op]
    white, black = (a, b) if a_white else (b, a)
    t0 = time.time()
    out = play_match(local(white, index), local(black, index), args.base_ms, args.increment_ms,
                     start_fen=fen)
    if out.result == "draw":
        a_score = 0.5
    elif out.result == "void":
        a_score = None
    else:
        a_score = 1.0 if (out.result == "white") == a_white else 0.0
    return {"index": index, "opening": name, "a_white": a_white, "white": white.name,
            "black": black.name, "result": out.result, "termination": out.termination,
            "a_score": a_score, "seconds": round(time.time() - t0, 1), "pgn": out.pgn}


def analyse(games: list[dict], elo0: float, elo1: float, alpha: float, beta: float,
            min_games: int = 0) -> dict:
    scored = [g for g in games if g["a_score"] is not None]
    w = sum(g["a_score"] == 1.0 for g in scored)
    d = sum(g["a_score"] == 0.5 for g in scored)
    l = sum(g["a_score"] == 0.0 for g in scored)
    by_index = {g["index"]: g for g in scored}
    pairs = [(by_index[i]["a_score"] + by_index[i + 1]["a_score"]) / 2.0
             for i in range(0, max(by_index, default=-1) + 1, 2) if i in by_index and i + 1 in by_index]
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    llr = gsprt_llr(pairs, elo0, elo1)
    res = {"games": len(games), "scored": len(scored), "wins": w, "draws": d, "losses": l,
           "pairs": len(pairs), "llr": round(llr, 3), "bounds": [round(lower, 3), round(upper, 3)],
           "elo0": elo0, "elo1": elo1, "verdict": "running"}
    if scored:
        m, lo, hi = interval([g["a_score"] for g in scored])
        res.update(score=round(m, 4), score_95=[round(lo, 4), round(hi, 4)], elo=round(score_to_elo(m), 1),
                   elo_95=[round(score_to_elo(lo), 1), round(score_to_elo(hi), 1)])
    if pairs:
        m, lo, hi = interval(pairs)
        res.update(pair_score=round(m, 4), pair_score_95=[round(lo, 4), round(hi, 4)])
    if len(scored) < min_games:
        pass
    elif llr >= upper:
        res["verdict"] = f"H1 accepted: A stronger than B (about {elo1:+g} Elo or more)"
    elif llr <= lower:
        res["verdict"] = f"H0 accepted: A not stronger than B (about {elo0:+g} Elo or less); see the interval for which way"
    term: dict[str, int] = {}
    fails = {"A": 0, "B": 0}
    for g in games:
        term[g["termination"]] = term.get(g["termination"], 0) + 1
        if g["termination"] in FAILED_TERMINATIONS:
            loser_is_a = g["a_score"] == 0.0
            fails["A" if loser_is_a else "B"] += 1
    res["terminations"] = term
    res["failures"] = fails
    return res


def run_match(spec: str, args: argparse.Namespace) -> dict:
    a_name, b_name, elo0, elo1 = spec.split(":")
    elo0, elo1 = float(elo0), float(elo1)
    a, b = ROOT / "engines" / a_name, ROOT / "engines" / b_name
    out_dir = ROOT / args.out / f"{a_name}_vs_{b_name}"
    (out_dir / "pgn").mkdir(parents=True, exist_ok=True)
    log = out_dir / "games.jsonl"
    games = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    done = {g["index"] for g in games}
    lock = threading.Lock()

    def summary(final: bool) -> dict:
        res = analyse(games, elo0, elo1, args.alpha, args.beta, args.min_games)
        if final and res["verdict"] == "running":
            if res["games"] >= args.max_games:
                res["verdict"] = f"inconclusive at {res['games']} games (cap {args.max_games})"
            else:
                res["verdict"] = f"stopped at {res['games']} games, before a verdict or the cap"
        workers = args.workers
        if args.report and (out_dir / "summary.json").exists():  # keep the workers the games were played with
            workers = json.loads((out_dir / "summary.json").read_text())["settings"]["workers"]
        res.update(match=f"{a_name} vs {b_name}", A=a_name, B=b_name,
                   hashes={a_name: sha256(a / "agent.py"), b_name: sha256(b / "agent.py")},
                   settings={"base_ms": args.base_ms, "increment_ms": args.increment_ms,
                             "max_games": args.max_games, "min_games": args.min_games, "alpha": args.alpha, "beta": args.beta,
                             "openings": len(OPENINGS), "workers": workers})
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
        return res

    print(f"\n=== {a_name} vs {b_name}  (H0 {elo0:+g} / H1 {elo1:+g} Elo, cap {args.max_games} games, "
          f"{len(games)} already played)", flush=True)
    res = summary(False)
    if args.report or res["verdict"] != "running":
        return summary(True)

    pending = [i for i in range(args.max_games) if i not in done]
    stop = False
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        running = {}
        while (pending and not stop) or running:
            while pending and not stop and len(running) < args.workers:
                i = pending.pop(0)
                running[pool.submit(play_one, i, a, b, args)] = i
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                running.pop(f)
                try:
                    g = f.result()
                except Exception as exc:  # harness error: record as void, keep going
                    print(f"  game failed in the harness: {exc!r}", flush=True)
                    continue
                with lock:
                    (out_dir / "pgn" / f"game-{g['index'] + 1:03d}.pgn").write_text(g["pgn"] + "\n")
                    g.pop("pgn")
                    games.append(g)
                    with log.open("a") as fh:
                        fh.write(json.dumps(g) + "\n")
                    res = summary(False)
                el = (time.time() - t_start) / 60
                print(f"  game {g['index'] + 1:3d} {g['opening'][:22]:22s} {g['white']:>4s}-{g['black']:<4s} "
                      f"{g['result']:5s} {g['termination']:14s} | {a_name} {res['wins']}-{res['draws']}-"
                      f"{res['losses']} score {res.get('score', 0):.3f} LLR {res['llr']:+.2f} "
                      f"[{res['bounds'][0]:+.2f},{res['bounds'][1]:+.2f}] {el:5.1f} min", flush=True)
                if res["verdict"] != "running" and not stop:
                    print(f"  -> {res['verdict']}; finishing games in flight", flush=True)
                    stop = True
            if os.path.exists(ROOT / "STOP"):
                stop = True
    return summary(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", default=DEFAULT_MATCHES, help="comma-separated A:B[:elo0:elo1]")
    ap.add_argument("--base-ms", type=int, default=10_000, help="harness arena default: 10 s")
    ap.add_argument("--increment-ms", type=int, default=100, help="harness arena default: 0.1 s")
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--min-games", type=int, default=40,
                    help="no stopping before this many games: the variance estimate needs data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 3))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    ap.add_argument("--report", action="store_true", help="print results so far without playing")
    args = ap.parse_args()

    specs = []
    for s in args.matches.split(","):
        parts = s.strip().split(":")
        if len(parts) == 2:
            parts += ["0", "35"]
        specs.append(":".join(parts))
    print(f"{len(OPENINGS)} start positions, {args.base_ms / 1000:g} s + {args.increment_ms / 1000:g} s, "
          f"{args.workers} games at once. Stop early with: touch STOP", flush=True)
    results = []
    for s in specs:
        results.append(run_match(s, args))
        if (ROOT / "STOP").exists():          # STOP ends the whole run, and is cleared for the next one
            (ROOT / "STOP").unlink()
            print("STOP file found: run ended after this match; rerun to resume", flush=True)
            break

    print("\n" + "=" * 100)
    for r in results:
        sc = f"{r.get('score', float('nan')):.3f} [{r.get('score_95', ['?', '?'])[0]}, {r.get('score_95', ['?', '?'])[1]}]"
        el = f"{r.get('elo', float('nan')):+.0f} [{r.get('elo_95', ['?', '?'])[0]}, {r.get('elo_95', ['?', '?'])[1]}]"
        print(f"{r['match']:12s} {r['wins']:3d}-{r['draws']:3d}-{r['losses']:3d}  score {sc}  Elo {el}  "
              f"LLR {r['llr']:+.2f}  failures {r['failures']}  -> {r['verdict']}")
    (ROOT / args.out / "all_summaries.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
