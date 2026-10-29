"""Search: static exchange evaluation, move ordering, quiescence, and
negamax with principal-variation search and a transposition table.

All mutable state lives in four buffers passed as arguments (u64, i32, i8,
i64); their layout is defined in config.py.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, objmode

from .config import *  # noqa: F401,F403
from .board import *  # noqa: F401,F403
from .hce import *  # noqa: F401,F403
from .nnue import *  # noqa: F401,F403

# ==========================================================================
# Static exchange evaluation
#
# A proper swap-off: each side brings its least valuable attacker, x-rays are
# picked up because the attacker set is recomputed against the shrinking
# occupancy, and either side may stop when continuing would lose material.
# ==========================================================================
@njit(cache=False)
def see(bb, mb, side, move, i64):
    fr = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7

    occ = bb[0] | bb[1]
    if flag == FLAG_EP:
        capsq = to - 8 if side == 0 else to + 8
        occ ^= SQ_BB[capsq]
        gain = PIECE_VALUE[PAWN]
    else:
        v = mb[to]
        gain = PIECE_VALUE[v] if v else 0

    on_sq = mb[fr]
    occ ^= SQ_BB[fr]
    if promo:
        gain += PIECE_VALUE[promo] - PIECE_VALUE[PAWN]
        on_sq = promo

    i64[SEE_OFF] = gain
    d = 0
    stm = 1 - side

    while d < 30:
        att = attackers_all(bb, to, occ) & bb[stm] & occ
        if not att:
            break
        chosen = -1
        chosen_pt = PAWN
        for pt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
            sub = att & bb[pt + 1]
            if sub:
                chosen = lsb(sub)
                chosen_pt = pt
                break
        if chosen < 0:
            break
        if chosen_pt == KING:
            rest = occ ^ SQ_BB[chosen]
            if attackers_all(bb, to, rest) & bb[1 - stm] & rest:
                break
        d += 1
        i64[SEE_OFF + d] = PIECE_VALUE[on_sq] - i64[SEE_OFF + d - 1]
        a = -i64[SEE_OFF + d]
        b = i64[SEE_OFF + d - 1]
        if (a if a > b else b) < 0:
            break
        occ ^= SQ_BB[chosen]
        on_sq = chosen_pt
        stm = 1 - stm

    while d > 0:
        a = -i64[SEE_OFF + d - 1]
        b = i64[SEE_OFF + d]
        i64[SEE_OFF + d - 1] = -(a if a > b else b)
        d -= 1
    return i64[SEE_OFF]


@njit(cache=False)
def is_losing_capture(bb, mb, side, move, i64):
    """Only a capture whose attacker outvalues its victim can lose the
    exchange, so equal and winning captures never pay for a SEE call."""
    to = (move >> 6) & 63
    flag = (move >> 15) & 7
    if flag == FLAG_EP:
        victim = PIECE_VALUE[PAWN]
    else:
        v = mb[to]
        victim = PIECE_VALUE[v] if v else PIECE_VALUE[PAWN]
    if PIECE_VALUE[mb[move & 63]] <= victim:
        return False
    return see(bb, mb, side, move, i64) < 0


# ==========================================================================
# Move ordering
# ==========================================================================
MVV_BONUS = 1_000_000
LOSING_CAPTURE_BASE = -50_000


@njit(cache=False)
def score_moves(bb, mb, side, i32, i64, base, n, tt_move, ply, prev_move, use_see):
    hbase = HIST_OFF + side * 4096
    cbase = COUNTER_OFF + side * 4096
    k0 = i32[KILLER_OFF + ply * 2]
    k1 = i32[KILLER_OFF + ply * 2 + 1]
    ckey = -1
    if prev_move:
        ckey = cbase + ((prev_move >> 6) & 63) * 64 + (prev_move & 63)
    for i in range(base, n):
        m = i32[i]
        if m == tt_move:
            i32[i + MSCORE_DELTA] = 2_000_000
            continue
        fr = m & 63
        to = (m >> 6) & 63
        flag = (m >> 15) & 7
        promo = (m >> 12) & 7
        if flag == FLAG_CAPTURE or flag == FLAG_EP or flag == FLAG_PROMO_CAP:
            if flag == FLAG_EP:
                victim = PIECE_VALUE[PAWN]
            else:
                v = mb[to]
                victim = PIECE_VALUE[v] if v else PIECE_VALUE[PAWN]
            if use_see and is_losing_capture(bb, mb, side, m, i64):
                i32[i + MSCORE_DELTA] = LOSING_CAPTURE_BASE + see(bb, mb, side, m, i64)
            else:
                i32[i + MSCORE_DELTA] = MVV_BONUS + victim * 10 - PIECE_VALUE[mb[fr]]
            continue
        if promo:
            i32[i + MSCORE_DELTA] = 900_000 + PIECE_VALUE[promo]
            continue
        if m == k0:
            i32[i + MSCORE_DELTA] = 800_000
            continue
        if m == k1:
            i32[i + MSCORE_DELTA] = 790_000
            continue
        s = i32[hbase + fr * 64 + to]
        if s > 7000:
            s = 7000
        if ckey >= 0 and i32[ckey] == m:
            s += 500
        i32[i + MSCORE_DELTA] = s


@njit(cache=False, inline="always")
def pick_move(i32, i, n):
    """Lazy selection: a cutoff after two moves does not pay to sort thirty."""
    best = i
    bs = i32[i + MSCORE_DELTA]
    for j in range(i + 1, n):
        if i32[j + MSCORE_DELTA] > bs:
            bs = i32[j + MSCORE_DELTA]
            best = j
    if best != i:
        tm = i32[i]
        i32[i] = i32[best]
        i32[best] = tm
        ts = i32[i + MSCORE_DELTA]
        i32[i + MSCORE_DELTA] = i32[best + MSCORE_DELTA]
        i32[best + MSCORE_DELTA] = ts
    return i32[i]


# ==========================================================================
# Clock
#
# time.time_ns() does not exist in nopython mode -- that is what killed the
# previous full-Numba attempt. objmode is supported; sampled once every 2048
# nodes its cost is unmeasurable.
# ==========================================================================
@njit(cache=False)
def now_ns():
    with objmode(t="int64"):
        t = time.perf_counter_ns()
    return t


@njit(cache=False, inline="always")
def tick(i64):
    i64[C_NODES] += 1
    # Halved from 2048. Nodes are no longer uniform in cost -- SEE and the
    # evaluation cache both make some nodes far more expensive than others --
    # and the deadline has to be sampled often enough to bound the worst one.
    if (i64[C_NODES] & 1023) == 0:
        if now_ns() > i64[C_DEADLINE]:
            i64[C_ABORT] = 1


# ==========================================================================
# Draws
# ==========================================================================
CONTEMPT_BASE = 30
DYNAMIC_CONTEMPT_MARGIN = 200


@njit(cache=False)
def draw_score(bb, mb, side, key, u64, i32, i64):
    static = evaluate(bb, mb, side, key, u64, i32)
    contempt = CONTEMPT_BASE
    if static > DYNAMIC_CONTEMPT_MARGIN:
        extra = static // 20
        contempt += extra if extra < 30 else 30
    elif static < -DYNAMIC_CONTEMPT_MARGIN:
        contempt = contempt - 20
        if contempt < 5:
            contempt = 5
    return -contempt if side == i64[C_ROOT] else contempt


@njit(cache=False)
def is_repetition(key, ply, u64, i64):
    for i in range(ply):
        if u64[PATH_OFF + i] == key:
            return True
    for i in range(i64[C_REPN]):
        if u64[REP_OFF + i] == key:
            return True
    return False


@njit(cache=False)
def insufficient(bb):
    if bb[PAWN + 1] or bb[ROOK + 1] or bb[QUEEN + 1]:
        return False
    return popcount(bb[KNIGHT + 1] | bb[BISHOP + 1]) <= 1


@njit(cache=False)
def has_non_pawn(bb, side):
    return ((bb[KNIGHT + 1] | bb[BISHOP + 1] | bb[ROOK + 1] | bb[QUEEN + 1])
            & bb[side]) != U0


# ==========================================================================
# Quiescence
# ==========================================================================
DELTA_MARGIN = 180


@njit(cache=False)
def quiescence(bb, mb, side, castling, ep, halfmove, key,
               alpha, beta, ply, qdepth, u64, i32, i8, i64):
    tick(i64)
    if i64[C_ABORT]:
        return 0
    if ply > i64[C_SELDEPTH]:
        i64[C_SELDEPTH] = ply
    if ply >= SEARCH_PLY_LIMIT:
        return evaluate(bb, mb, side, key, u64, i32)

    alpha_original = alpha
    idx = np.int64(key & TT_MASK)
    tt_move = NO_MOVE
    if USE_QS_TT and u64[idx] == key and i8[idx] >= 0:
        # Any stored entry was searched at least as hard as a quiescence node,
        # so no depth test is needed here, only the mate-distance correction.
        sc = np.int64(i32[idx])
        if sc > MATE_BOUND:
            sc -= ply
        elif sc < -MATE_BOUND:
            sc += ply
        fl = i8[TT_FLAG_OFF + idx]
        if fl == TT_EXACT:
            return sc
        if fl == TT_LOWER and sc >= beta:
            return sc
        if fl == TT_UPPER and sc <= alpha:
            return sc
        tt_move = i32[TT_MOVE_OFF + idx]

    checked = in_check(bb, side)
    base = MOVES_OFF + ply * MAX_MOVES

    if checked:
        n = gen_legal(bb, mb, side, castling, ep, i32, base, False)
        if n == base:
            return -MATE_SCORE + ply
        if qdepth >= MAX_QDEPTH:
            return evaluate(bb, mb, side, key, u64, i32)
        best = -INFINITY
    else:
        best = evaluate(bb, mb, side, key, u64, i32)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        if qdepth >= MAX_QDEPTH:
            return best
        n = gen_legal(bb, mb, side, castling, ep, i32, base, True)
        if n == base:
            return best
        # Losing captures and hopeless deltas are filtered before ordering,
        # so the sort never ranks a move that is then discarded.
        w = base
        for i in range(base, n):
            m = i32[i]
            if ((m >> 12) & 7) == 0:
                v = mb[(m >> 6) & 63]
                gain = PIECE_VALUE[v] if v else PIECE_VALUE[PAWN]
                if best + gain + DELTA_MARGIN < alpha:
                    continue
                if is_losing_capture(bb, mb, side, m, i64):
                    continue
            i32[w] = m
            w += 1
        n = w
        if n == base:
            return best

    score_moves(bb, mb, side, i32, i64, base, n, tt_move, ply, NO_MOVE, False)

    best_move = NO_MOVE
    for i in range(base, n):
        m = pick_move(i32, i, n)
        # Repeated on purpose: best and alpha both rise as moves are searched,
        # so the in-loop test prunes strictly more than the pre-pass could.
        if not checked and ((m >> 12) & 7) == 0:
            v = mb[(m >> 6) & 63]
            gain = PIECE_VALUE[v] if v else PIECE_VALUE[PAWN]
            if best + gain + DELTA_MARGIN < alpha:
                continue
        c2, e2, h2, k2 = make_move(bb, mb, side, castling, ep, halfmove, key,
                                   m, i8, UNDO_OFF + ply)
        sc = -quiescence(bb, mb, 1 - side, c2, e2, h2, k2, -beta, -alpha,
                         ply + 1, qdepth + 1, u64, i32, i8, i64)
        unmake_move(bb, mb, side, m, i8, UNDO_OFF + ply)
        if i64[C_ABORT]:
            return 0
        if sc > best:
            best = sc
            best_move = m
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    # Deliberately stricter than the main-search replacement policy: a depth-0
    # entry may take an empty slot, refresh its own key, or overwrite another
    # depth-0 entry, but it never evicts a deep one. Without that a quiescence
    # store would clear out the table it is meant to be helping.
    if USE_QS_TT and best > -INFINITY:
        if i8[idx] < 0 or u64[idx] == key or i8[idx] == 0:
            stored = best
            if stored > MATE_BOUND:
                stored += ply
            elif stored < -MATE_BOUND:
                stored -= ply
            if best <= alpha_original:
                fl = TT_UPPER
            elif best >= beta:
                fl = TT_LOWER
            else:
                fl = TT_EXACT
            u64[idx] = key
            i32[idx] = stored
            i32[TT_MOVE_OFF + idx] = best_move
            i8[idx] = 0
            i8[TT_FLAG_OFF + idx] = fl
    return best


# ==========================================================================
# Main search
# ==========================================================================
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 3
IID_MIN_DEPTH = 5
IID_REDUCTION = 2
NULL_BASE_R = 2
EXTENSION_PLY_LIMIT = 48

# Reverse futility: at shallow depth a static score this far above beta is
# taken at face value rather than searched. The margin shrinks by one depth
# unit when the side to move is improving, because a rising score is less
# likely to be an artefact of the horizon.
RFP_MAX_DEPTH = 8
RFP_MARGIN = 80

# Late move pruning. Indexed [improving, depth]; a quiet move at or past the
# threshold is skipped outright. The not-improving row is roughly half the
# improving row, which is the usual shape.
LMP_MAX_DEPTH = 8
LMP_TABLE = np.zeros((2, LMP_MAX_DEPTH + 1), dtype=np.int64)
for _d in range(LMP_MAX_DEPTH + 1):
    LMP_TABLE[1, _d] = 3 + _d * _d
    LMP_TABLE[0, _d] = (3 + _d * _d) // 2

# Late move reductions. numbav1 capped this at two plies; the log form reaches
# four at depth 12 and five at depth 20, which is where the branching factor
# actually comes down. The full-depth verification re-search below is what
# keeps it from hiding two-ply refutations.
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _i in range(1, 64):
        LMR_TABLE[_d, _i] = int(0.75 + math.log(_d) * math.log(_i) / 2.25)

# SEE pruning in the main search. Captures tolerate a larger loss than quiets
# because a losing capture is often still the move that keeps a line alive.
SEE_PRUNE_DEPTH = 6
SEE_CAPTURE_MARGIN = -90
SEE_QUIET_MARGIN = -50


@njit(cache=False)
def negamax(bb, mb, side, castling, ep, halfmove, key, depth, alpha, beta,
            ply, prev_move, u64, i32, i8, i64):
    tick(i64)
    if i64[C_ABORT]:
        return 0
    if ply > i64[C_SELDEPTH]:
        i64[C_SELDEPTH] = ply
    if ply >= SEARCH_PLY_LIMIT:
        return evaluate(bb, mb, side, key, u64, i32)

    if ply > 0:
        if (is_repetition(key, ply, u64, i64) or halfmove >= 100
                or insufficient(bb)):
            return draw_score(bb, mb, side, key, u64, i32, i64)
        lo = -MATE_SCORE + ply
        hi = MATE_SCORE - ply - 1
        if alpha < lo:
            alpha = lo
        if beta > hi:
            beta = hi
        if alpha >= beta:
            return alpha

    alpha_original = alpha
    idx = np.int64(key & TT_MASK)
    tt_move = NO_MOVE
    if u64[idx] == key and i8[idx] >= 0:
        tt_move = i32[TT_MOVE_OFF + idx]
        if i8[idx] >= depth and ply > 0:
            sc = np.int64(i32[idx])
            if sc > MATE_BOUND:
                sc -= ply
            elif sc < -MATE_BOUND:
                sc += ply
            fl = i8[TT_FLAG_OFF + idx]
            if fl == TT_EXACT:
                return sc
            if fl == TT_LOWER and sc >= beta:
                return sc
            if fl == TT_UPPER and sc <= alpha:
                return sc

    checked = in_check(bb, side)
    if checked and ply < EXTENSION_PLY_LIMIT:
        depth += 1

    if depth <= 0:
        return quiescence(bb, mb, side, castling, ep, halfmove, key,
                          alpha, beta, ply, 0, u64, i32, i8, i64)

    is_pv = beta - alpha > 1
    static_eval = 0
    improving = 1
    if checked:
        # No usable static score in check, so the trend is unknown. Assuming
        # improving is the conservative choice: it prunes less.
        i32[EVSTACK_OFF + ply] = NO_EVAL
    else:
        static_eval = evaluate(bb, mb, side, key, u64, i32)
        i32[EVSTACK_OFF + ply] = static_eval
        if ply >= 2 and i32[EVSTACK_OFF + ply - 2] != NO_EVAL:
            if static_eval <= i32[EVSTACK_OFF + ply - 2]:
                improving = 0

        # Reverse futility. A score this far clear of beta at shallow depth is
        # not going to be talked out of it by a search, and searching it is
        # what the branching factor is spent on.
        if (USE_RFP and depth <= RFP_MAX_DEPTH and not is_pv
                and beta < MATE_BOUND and beta > -MATE_BOUND
                and has_non_pawn(bb, side)
                and static_eval - RFP_MARGIN * (depth - improving) >= beta):
            return static_eval

        # Razoring: trust the quiescence score only when it confirms the
        # position really is below alpha.
        if depth <= 3 and static_eval + depth * 70 < alpha:
            q = quiescence(bb, mb, side, castling, ep, halfmove, key,
                           alpha, beta, ply, 0, u64, i32, i8, i64)
            if i64[C_ABORT]:
                return 0
            if q < alpha:
                return q

        # Null move. has_non_pawn excludes the king, so this cannot fire in a
        # king-and-pawn endgame where zugzwang makes it unsound.
        if (depth >= 3 and beta < MATE_BOUND and static_eval >= beta
                and has_non_pawn(bb, side)):
            r = NULL_BASE_R + depth // 6
            if USE_NULL_ADAPTIVE:
                extra = (static_eval - beta) // 200
                if extra > 3:
                    extra = 3
                if extra > 0:
                    r += extra
            nkey = key ^ Z_SIDE
            if ep >= 0:
                nkey ^= Z_EP_FILE[ep & 7]
            u64[PATH_OFF + ply] = key
            sc = -negamax(bb, mb, 1 - side, castling, -1, halfmove + 1, nkey,
                          depth - 1 - r, -beta, -beta + 1, ply + 1, NO_MOVE,
                          u64, i32, i8, i64)
            if i64[C_ABORT]:
                return 0
            if sc >= beta:
                if sc > MATE_BOUND:
                    return beta
                return sc

    # Internal iterative deepening: with no table move, ordering at a PV node
    # is close to random, and ordering is what alpha-beta pays for.
    if tt_move == NO_MOVE and depth >= IID_MIN_DEPTH and beta - alpha > 1 and not checked:
        negamax(bb, mb, side, castling, ep, halfmove, key,
                depth - IID_REDUCTION, alpha, beta, ply, prev_move,
                u64, i32, i8, i64)
        if i64[C_ABORT]:
            return 0
        if u64[idx] == key:
            tt_move = i32[TT_MOVE_OFF + idx]

    base = MOVES_OFF + ply * MAX_MOVES
    n = gen_legal(bb, mb, side, castling, ep, i32, base, False)
    if n == base:
        if checked:
            return -MATE_SCORE + ply
        return draw_score(bb, mb, side, key, u64, i32, i64)

    score_moves(bb, mb, side, i32, i64, base, n, tt_move, ply, prev_move, True)
    u64[PATH_OFF + ply] = key

    hbase = HIST_OFF + side * 4096
    cbase = COUNTER_OFF + side * 4096
    best = -INFINITY
    best_move = NO_MOVE

    for i in range(base, n):
        m = pick_move(i32, i, n)
        mscore = i32[i + MSCORE_DELTA]
        index = i - base
        fr = m & 63
        to = (m >> 6) & 63
        flag = (m >> 15) & 7
        is_quiet = (flag == FLAG_QUIET or flag == FLAG_CASTLE or flag == FLAG_DOUBLE)

        if (not checked and depth <= 3 and is_quiet and index >= 2
                and static_eval + 100 * depth < alpha):
            continue

        # Late move pruning. Once enough moves have been tried at shallow
        # depth, the remainder are quiet moves the ordering has already ranked
        # as unpromising; searching them is what kept numbav1 at depth 12.
        # best > -MATE_BOUND guarantees a legal move has already been scored,
        # so this can never leave the node without a move.
        if (USE_LMP and not checked and not is_pv and is_quiet
                and depth <= LMP_MAX_DEPTH and best > -MATE_BOUND
                and index >= LMP_TABLE[improving, depth]):
            continue

        # SEE pruning, rebuilt. numbav2 called see() for every move here,
        # which is what pushed the node cost past what the deadline sampler
        # could bound.
        #
        # Captures cost nothing now: score_moves already ran SEE on every
        # capture whose attacker outvalues its victim and stored
        # LOSING_CAPTURE_BASE + see in the ordering score. pick_move swaps the
        # score alongside the move, so mscore is this move's score, and only a
        # losing capture is ever negative -- quiets are history values, which
        # are clamped non-negative.
        if (USE_SEE_PRUNE_CAPTURE and not checked and not is_quiet
                and depth <= SEE_PRUNE_DEPTH and index > 0
                and best > -MATE_BOUND and mscore < 0
                and mscore - LOSING_CAPTURE_BASE < SEE_CAPTURE_MARGIN * depth):
            continue

        # Quiets go through a filter first. The case worth pruning is a piece
        # stepping onto a square an enemy pawn covers; that is one bitboard
        # test, and it removes almost every swap-off that used to run here.
        #
        # Castling is excluded because FLAG_CASTLE is quiet, and see() on it
        # reads a king worth 10000 moving to its castled square with the rook
        # still home. The swap list is meaningless and castling gets pruned
        # outright. That is what crashed numbav2. King and pawn moves are out
        # for the same reason: PIECE_VALUE[KING] makes the result noise, and a
        # pawn cannot lose the exchange to a pawn.
        if (USE_SEE_PRUNE_QUIET and not checked and is_quiet
                and flag != FLAG_CASTLE and depth <= SEE_PRUNE_DEPTH
                and index > 0 and best > -MATE_BOUND):
            mover = mb[fr]
            if mover != PAWN and mover != KING:
                if PAWN_ATT[side, to] & bb[PAWN + 1] & bb[1 - side]:
                    if see(bb, mb, side, m, i64) < SEE_QUIET_MARGIN * depth:
                        continue

        c2, e2, h2, k2 = make_move(bb, mb, side, castling, ep, halfmove, key,
                                   m, i8, UNDO_OFF + ply)
        gives_check = in_check(bb, 1 - side)

        reduction = 0
        if (depth >= LMR_MIN_DEPTH and index >= LMR_MIN_MOVE and is_quiet
                and not checked and not gives_check):
            if USE_LMR_TABLE:
                d_i = depth if depth < 63 else 63
                m_i = index if index < 63 else 63
                reduction = LMR_TABLE[d_i, m_i]
                # A principal-variation node is the one place a wrong
                # reduction is not recoverable, so it reduces one less.
                if is_pv:
                    reduction -= 1
                if improving == 0:
                    reduction += 1
                if (m == i32[KILLER_OFF + ply * 2]
                        or m == i32[KILLER_OFF + ply * 2 + 1]):
                    reduction -= 1
                h = i32[hbase + fr * 64 + to]
                if h > 4000:
                    reduction -= 1
                elif h < 100:
                    reduction += 1
            else:
                reduction = 1
                if index >= 6 and depth >= 5:
                    reduction += 1
                if i32[hbase + fr * 64 + to] > 1000:
                    reduction -= 1
            if reduction < 0:
                reduction = 0
            if reduction > depth - 2:
                reduction = depth - 2
                if reduction < 0:
                    reduction = 0

        nd = depth - 1
        if index == 0:
            sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, nd,
                          -beta, -alpha, ply + 1, m, u64, i32, i8, i64)
        else:
            sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, nd - reduction,
                          -alpha - 1, -alpha, ply + 1, m, u64, i32, i8, i64)
            if not i64[C_ABORT] and reduction and sc > alpha:
                sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, nd,
                              -alpha - 1, -alpha, ply + 1, m, u64, i32, i8, i64)
            if not i64[C_ABORT] and alpha < sc < beta:
                sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, nd,
                              -beta, -alpha, ply + 1, m, u64, i32, i8, i64)
        unmake_move(bb, mb, side, m, i8, UNDO_OFF + ply)

        if i64[C_ABORT]:
            return 0

        if sc > best:
            best = sc
            best_move = m
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if is_quiet:
                if i32[KILLER_OFF + ply * 2] != m:
                    i32[KILLER_OFF + ply * 2 + 1] = i32[KILLER_OFF + ply * 2]
                    i32[KILLER_OFF + ply * 2] = m
                h = i32[hbase + fr * 64 + to] + depth * depth
                if h > 1 << 20:
                    h = 1 << 20
                i32[hbase + fr * 64 + to] = h
                if prev_move:
                    i32[cbase + ((prev_move >> 6) & 63) * 64 + (prev_move & 63)] = m
            break

    if i8[idx] < 0 or u64[idx] != key or i8[idx] <= depth:
        stored = best
        if stored > MATE_BOUND:
            stored += ply
        elif stored < -MATE_BOUND:
            stored -= ply
        if best <= alpha_original:
            fl = TT_UPPER
        elif best >= beta:
            fl = TT_LOWER
        else:
            fl = TT_EXACT
        u64[idx] = key
        i32[idx] = stored
        i32[TT_MOVE_OFF + idx] = best_move
        i8[idx] = depth if depth < 127 else 127
        i8[TT_FLAG_OFF + idx] = fl

    return best


@njit(cache=False)
def search_root(bb, mb, side, castling, ep, halfmove, key, depth,
                alpha, beta, previous_best, u64, i32, i8, i64):
    base = MOVES_OFF
    n = gen_legal(bb, mb, side, castling, ep, i32, base, False)
    if n == base:
        return NO_MOVE, 0
    score_moves(bb, mb, side, i32, i64, base, n, previous_best, 0, NO_MOVE, True)
    u64[PATH_OFF] = key

    best_move = i32[base]
    best_score = -INFINITY

    for i in range(base, n):
        m = pick_move(i32, i, n)
        c2, e2, h2, k2 = make_move(bb, mb, side, castling, ep, halfmove, key,
                                   m, i8, UNDO_OFF)
        if i == base:
            sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, depth - 1,
                          -beta, -alpha, 1, m, u64, i32, i8, i64)
        else:
            sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, depth - 1,
                          -alpha - 1, -alpha, 1, m, u64, i32, i8, i64)
            if not i64[C_ABORT] and alpha < sc < beta:
                sc = -negamax(bb, mb, 1 - side, c2, e2, h2, k2, depth - 1,
                              -beta, -alpha, 1, m, u64, i32, i8, i64)
        unmake_move(bb, mb, side, m, i8, UNDO_OFF)
        if i64[C_ABORT]:
            return best_move, best_score
        if sc > best_score:
            best_score = sc
            best_move = m
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            break
    return best_move, best_score


@njit(cache=False)
def decay_history(i32):
    """History is halved each move so old data fades; killers are cleared
    because they are indexed by ply and the ply numbering shifts by two."""
    for i in range(HIST_OFF, COUNTER_OFF):
        i32[i] //= 2
    for i in range(KILLER_OFF, KILLER_OFF + 2 * MAX_PLY):
        i32[i] = 0
