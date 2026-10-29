"""Entry point: get_move(fen, time_left_ms) -> UCI move.

Time management, iterative deepening with aspiration windows, repetition
tracking between moves, and the warm-up that compiles every kernel at import.
"""
from __future__ import annotations

import time

import chess
import numpy as np

from .config import *  # noqa: F401,F403
from .board import *  # noqa: F401,F403
from .hce import *  # noqa: F401,F403
from .nnue import *  # noqa: F401,F403
from .search import *  # noqa: F401,F403

U64, I32, I8, I64 = new_state()


# ==========================================================================
# Time management (v10 constants, including the v10 "D" cap-ordering fix:
# the critical multiplier is applied to the raw allocation, before the share
# caps, so SOFT_SHARE_CAP and HARD_SHARE_CAP actually bound the move)
# ==========================================================================
MOVE_OVERHEAD_MS = 400.0 if SAFE_CLOCK else 250.0
# Rated games are 0.5 s per move, and numbav1 hard-coded that. Every move's
# budget therefore assumed a 375 ms refund. In a rated game that is right; in
# an arena game at a shorter time control it is 375 ms per move that never
# comes back, and the clock bleeds until it flags. The referee hands us
# time_left_ms every move, so the increment is measurable rather than assumed:
# inc = left_now - left_before + used_before. Our own timing runs slightly
# long against the referee's wall clock, so the estimate errs low, which is
# the safe direction. The running minimum is kept for the same reason.
INCREMENT_MS = 500.0
INCREMENT_USE = 0.75
SOFT_SHARE_CAP = 0.40
HARD_MULTIPLIER = 2.5
HARD_SHARE_CAP = 0.45 if SAFE_CLOCK else 0.55
NEXT_DEPTH_FRACTION = 0.65
MIN_BUDGET_MS = 25.0
CRITICAL_MULTIPLIER = 1.5
MAX_DEPTH = 64
ASPIRATION_WINDOW = 40
ASPIRATION_MAX = 800
# A window failure means the root moved, which is worth extra time -- but in
# numbav1 "extra" meant the hard cap, and three moves across the eight rated
# games cost 8.7 s, 8.2 s and 7.6 s that way. This bounds the re-search chain
# at one and a half soft budgets instead of two and a half.
ASPIRATION_EXTEND = 1.5
# Once a mating line is in hand and cannot be resolved inside the depth
# searched, fall back on the move that has survived this many iterations --
# but only after MATE_SOFT_SHARE of the soft budget has actually been spent
# looking for a shorter one. numbav6 had no time condition, so three cheap
# iterations at depth four were enough to act on a mate distance inherited
# whole from the table. See the round-86 note in the header.
MATE_STABLE_ITERS = 3
# A mate score is only acted on once the mate is short enough to have been
# resolved inside the depth actually searched, and never below this depth.
MATE_CONFIRM_DEPTH = 4
# Fraction of the soft budget an unresolved mate may consume before the
# stable-move fallback is allowed to fire. numbav2 effectively used 1.0 here
# and flagged; numbav6 effectively used 0.0 and shuffled.
MATE_SOFT_SHARE = 0.50


def budget_ms(time_left_ms: int, is_critical: bool, piece_count: int,
              increment_ms: float = INCREMENT_MS):
    usable = max(0.0, float(time_left_ms) - MOVE_OVERHEAD_MS)
    moves_to_go = 10.0 + (piece_count - 2) / 30.0 * 30.0
    if moves_to_go < 10.0:
        moves_to_go = 10.0
    soft = usable / moves_to_go + increment_ms * INCREMENT_USE
    if is_critical:
        soft *= CRITICAL_MULTIPLIER
    soft = min(soft, usable * SOFT_SHARE_CAP)
    hard = min(soft * HARD_MULTIPLIER, usable * HARD_SHARE_CAP)
    soft = max(MIN_BUDGET_MS, soft)
    hard = max(soft, hard)
    return soft, hard


# ==========================================================================
# Game state that survives between moves
# ==========================================================================
_SEEN: dict = {}
_STARTED = [False]
VERBOSE = True

# [previous time_left_ms, milliseconds we spent on it, increment estimate]
_CLOCK = [-1.0, 0.0, INCREMENT_MS]


def _observe_increment(time_left_ms: int) -> float:
    if not ADAPTIVE_INCREMENT:
        return INCREMENT_MS
    if _CLOCK[0] >= 0.0:
        est = float(time_left_ms) - _CLOCK[0] + _CLOCK[1]
        if 0.0 <= est < 5000.0 and est < _CLOCK[2]:
            _CLOCK[2] = est
    return _CLOCK[2]


def _record_clock(time_left_ms: int, used_ms: float) -> None:
    _CLOCK[0] = float(time_left_ms)
    _CLOCK[1] = used_ms


def _load_repetitions(u64, i64) -> None:
    """Positions that have already occurred twice in the game: the one now on
    the board would make the third, so the search may claim the draw."""
    n = 0
    for k, c in _SEEN.items():
        if c >= 2 and n < REP_CAP:
            u64[REP_OFF + n] = np.uint64(k)
            n += 1
    i64[C_REPN] = n


def _search_move(fen: str, time_left_ms: int) -> str:
    started = time.perf_counter()

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    fallback = legal[0].uci()

    bb, mb, side, castling, ep, halfmove = fen_to_state(fen)
    key = compute_key(bb, mb, side, castling, ep)
    _SEEN[int(key)] = _SEEN.get(int(key), 0) + 1

    if len(legal) == 1:
        # numbav1 pushed a null move here and never recorded the result, so a
        # forced sequence punched a hole in the threefold history the search
        # relies on. Generate the move properly and record the position.
        if USE_FORCED_MOVE_REPFIX:
            n = gen_legal(bb, mb, side, castling, ep, I32, MOVES_OFF, False)
            if n == MOVES_OFF + 1:
                m = int(I32[MOVES_OFF])
                uci = move_to_uci(m)
                if uci == fallback:
                    c2, e2, h2, k2 = make_move(bb, mb, side, castling, ep,
                                               halfmove, key, m, I8, UNDO_OFF)
                    _SEEN[int(k2)] = _SEEN.get(int(k2), 0) + 1
        _record_clock(time_left_ms, (time.perf_counter() - started) * 1000.0)
        return fallback

    _load_repetitions(U64, I64)

    piece_count = int(popcount(bb[0] | bb[1]))
    soft, hard = budget_ms(time_left_ms, board.is_check(), piece_count,
                           _observe_increment(time_left_ms))

    I64[C_ROOT] = side
    I64[C_NODES] = 0
    I64[C_SELDEPTH] = 0
    I64[C_ABORT] = 0
    I64[C_DEADLINE] = time.perf_counter_ns() + int(hard * 1_000_000.0)
    decay_history(I32)
    # Plies 0 and 1 are never written by negamax, so the improving test at
    # ply 2 would otherwise read whatever the previous move left behind.
    I32[EVSTACK_OFF] = NO_EVAL
    I32[EVSTACK_OFF + 1] = NO_EVAL

    best_move = 0
    score = 0
    depth_reached = 0
    mate_move = 0
    mate_stable = 0

    try:
        for depth in range(1, MAX_DEPTH + 1):
            if depth > 1 and (time.perf_counter() - started) * 1000.0 > soft * NEXT_DEPTH_FRACTION:
                break

            accept = True
            if depth <= 3 or abs(score) > MATE_BOUND:
                mv, sc = search_root(bb, mb, side, castling, ep, halfmove, key,
                                     depth, -INFINITY, INFINITY, best_move,
                                     U64, I32, I8, I64)
            else:
                # Widen only the side that failed. Widening both throws away
                # the bound the search had already established.
                window = ASPIRATION_WINDOW
                alpha = score - window
                beta = score + window
                while True:
                    mv, sc = search_root(bb, mb, side, castling, ep, halfmove,
                                         key, depth, alpha, beta, best_move,
                                         U64, I32, I8, I64)
                    if I64[C_ABORT] or alpha < sc < beta:
                        break
                    if (USE_ASPIRATION_TIME_GUARD
                            and (time.perf_counter() - started) * 1000.0
                            > soft * ASPIRATION_EXTEND):
                        # A fail high still bounds the move from below, so it
                        # is safe to keep. A fail low does not, so the
                        # previous iteration's move stands.
                        if sc <= alpha:
                            accept = False
                        break
                    if window > ASPIRATION_MAX:
                        mv, sc = search_root(bb, mb, side, castling, ep,
                                             halfmove, key, depth,
                                             -INFINITY, INFINITY, mv,
                                             U64, I32, I8, I64)
                        break
                    window *= 4
                    if sc <= alpha:
                        beta = (alpha + beta) // 2
                        alpha = sc - window
                    else:
                        beta = sc + window

            if I64[C_ABORT]:
                break
            if mv and accept:
                best_move = int(mv)
                score = int(sc)
                depth_reached = depth
            if not accept:
                break
            if abs(score) > MATE_BOUND:
                # numbav1 stopped here unconditionally: at depth 1 the root
                # inherits a mate score straight from the table and plays a
                # 50-node move with no verification, which is how the round-58
                # game spent 37 moves shuffling after finding mate. numbav2
                # over-corrected and chased the mate distance to depth 25.
                # Stability is the cheap middle: reaching depth four rules out
                # the shallow table hit, and three iterations agreeing on the
                # same move rules out the oscillation, both for a few cheap
                # iterations rather than the whole budget.
                if not USE_MATE_DEPTH_GUARD:
                    break
                if best_move == mate_move:
                    mate_stable += 1
                else:
                    mate_move = best_move
                    mate_stable = 1
                # Resolved: the whole mate fits inside the depth just
                # searched, so the distance is real and the move is the
                # shortest one. Nothing deeper can improve it.
                if MATE_SCORE - abs(score) <= depth:
                    break
                # Not resolved. numbav6 broke here anyway the moment three
                # iterations agreed, which at depth four means acting on a
                # mate score lifted whole out of the table with nothing
                # verified beneath it. Every mating move scores the same
                # under those conditions, so the choice between them is
                # arbitrary and can loop. Keep deepening instead: a deeper
                # search prefers the shorter mate because its score is
                # strictly higher. The time condition is what stops this
                # becoming the numbav2 failure of burning the cap on every
                # move of a won game.
                if (mate_stable >= MATE_STABLE_ITERS
                        and depth >= MATE_CONFIRM_DEPTH
                        and (time.perf_counter() - started) * 1000.0
                        > soft * MATE_SOFT_SHARE):
                    break
            else:
                mate_stable = 0
                mate_move = 0
    except Exception as error:  # pragma: no cover - defensive only
        print(f"search failed, playing fallback: {error!r}")

    if not best_move:
        _record_clock(time_left_ms, (time.perf_counter() - started) * 1000.0)
        return fallback

    uci = move_to_uci(best_move)
    # Final legality guard, outside the search. It should never reject a
    # correctly generated move; it stops a programming error from becoming an
    # illegal submission.
    if uci not in {m.uci() for m in legal}:
        print(f"generated illegal move {uci!r}, playing fallback")
        _record_clock(time_left_ms, (time.perf_counter() - started) * 1000.0)
        return fallback

    elapsed = (time.perf_counter() - started) * 1000.0
    _record_clock(time_left_ms, elapsed)
    if VERBOSE:
        print(f"d{depth_reached}/{I64[C_SELDEPTH]} {uci} cp{score} "
              f"n{I64[C_NODES]} {elapsed:.0f}ms "
              f"(soft {soft:.0f} hard {hard:.0f} inc {_CLOCK[2]:.0f}, "
              f"left {time_left_ms})")

    # Record the position handed back to the opponent, so threefold detection
    # sees both sides of a shuffle.
    c2, e2, h2, k2 = make_move(bb, mb, side, castling, ep, halfmove,
                               key, best_move, I8, UNDO_OFF)
    _SEEN[int(k2)] = _SEEN.get(int(k2), 0) + 1
    return uci


def get_move(fen: str, time_left_ms: int) -> str:
    """Exception guard around the search.

    An exception escaping this function loses the game outright, so nothing is
    allowed to escape. numbav2's crashes came from inside the jitted search
    rather than from Python, which this cannot catch -- but the forced-move
    path and the FEN parsing added in v2 are Python, and they were outside any
    guard.
    """
    try:
        return _search_move(fen, time_left_ms)
    except Exception as error:  # pragma: no cover - defensive only
        print(f"get_move failed, playing fallback: {error!r}")
        try:
            legal = list(chess.Board(fen).legal_moves)
            return legal[0].uci() if legal else "0000"
        except Exception:
            return "0000"


# ==========================================================================
# Warm-up: every kernel compiles inside the import budget, never on the clock
# ==========================================================================
def warmup() -> None:
    global VERBOSE
    VERBOSE = False
    try:
        for fen in (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        ):
            get_move(fen, 350)
    finally:
        VERBOSE = True
    _SEEN.clear()
    _CLOCK[0] = -1.0
    _CLOCK[1] = 0.0
    _CLOCK[2] = INCREMENT_MS
    U64[:] = 0
    I32[:] = 0
    I8[:] = 0
    I8[:TT_SIZE] = -1
    I64[:] = 0


# Warm-up runs last: the weights have to be in place before the kernels
# compile, because numba resolves a global array when it compiles the
# function that reads it.
try:
    warmup()
except Exception as exc:  # pragma: no cover
    # Never swallowed: if compilation fails the validation log has to say so.
    print(f"WARMUP FAILED: {exc!r}")
