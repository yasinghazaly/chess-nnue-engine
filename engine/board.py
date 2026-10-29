"""Board representation: bitboard geometry, Zobrist keys, attack queries,
legal move generation, make/unmake, and FEN/UCI conversion.

Position state used throughout the engine:
    bb   uint64[8]  bb[0], bb[1] = White, Black occupancy; bb[piece + 1] = that piece type
    mb   int8[64]   piece type on each square (0 = empty), a1 = 0, h8 = 63
    side int        0 = White to move, 1 = Black
"""
from __future__ import annotations

import chess
import numpy as np
from numba import njit

from .config import *  # noqa: F401,F403

# ==========================================================================
# Precomputed geometry
# ==========================================================================
SQ_BB = np.array([1 << s for s in range(64)], dtype=np.uint64)

INDEX64 = np.zeros(64, dtype=np.int64)
_DBJ = int(DEBRUIJN)
for _i in range(64):
    INDEX64[(((1 << _i) * _DBJ) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i

FILE_BB = np.zeros(8, dtype=np.uint64)
RANK_BB = np.zeros(8, dtype=np.uint64)
for _s in range(64):
    FILE_BB[_s & 7] |= np.uint64(1 << _s)
    RANK_BB[_s >> 3] |= np.uint64(1 << _s)

ADJ_FILE = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    if _f > 0:
        ADJ_FILE[_f] |= FILE_BB[_f - 1]
    if _f < 7:
        ADJ_FILE[_f] |= FILE_BB[_f + 1]

KNIGHT_ATT = np.zeros(64, dtype=np.uint64)
KING_ATT = np.zeros(64, dtype=np.uint64)
PAWN_ATT = np.zeros((2, 64), dtype=np.uint64)
for _s in range(64):
    _r, _f = divmod(_s, 8)
    for _dr, _df in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                     (1, 2), (1, -2), (-1, 2), (-1, -2)):
        _rr, _ff = _r + _dr, _f + _df
        if 0 <= _rr < 8 and 0 <= _ff < 8:
            KNIGHT_ATT[_s] |= np.uint64(1 << (_rr * 8 + _ff))
    for _dr in (-1, 0, 1):
        for _df in (-1, 0, 1):
            if _dr or _df:
                _rr, _ff = _r + _dr, _f + _df
                if 0 <= _rr < 8 and 0 <= _ff < 8:
                    KING_ATT[_s] |= np.uint64(1 << (_rr * 8 + _ff))
    for _df in (-1, 1):
        _ff = _f + _df
        if 0 <= _ff < 8:
            if _r + 1 < 8:
                PAWN_ATT[0, _s] |= np.uint64(1 << ((_r + 1) * 8 + _ff))
            if _r - 1 >= 0:
                PAWN_ATT[1, _s] |= np.uint64(1 << ((_r - 1) * 8 + _ff))

# Rays: 0=N 1=NE 2=E 3=SE 4=S 5=SW 6=W 7=NW. 0,1,2,7 scan forward from lsb.
DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
RAY = np.zeros((8, 64), dtype=np.uint64)
for _d, (_dr, _df) in enumerate(DIRS):
    for _s in range(64):
        _r, _f = divmod(_s, 8)
        _rr, _ff = _r + _dr, _f + _df
        while 0 <= _rr < 8 and 0 <= _ff < 8:
            RAY[_d, _s] |= np.uint64(1 << (_rr * 8 + _ff))
            _rr += _dr
            _ff += _df

BISHOP_PSEUDO = RAY[1] | RAY[3] | RAY[5] | RAY[7]
ROOK_PSEUDO = RAY[0] | RAY[2] | RAY[4] | RAY[6]

BETWEEN = np.zeros(64 * 64, dtype=np.uint64)
LINE = np.zeros(64 * 64, dtype=np.uint64)
for _a in range(64):
    for _d in range(8):
        _r, _f = divmod(_a, 8)
        _dr, _df = DIRS[_d]
        _acc = 0
        _rr, _ff = _r + _dr, _f + _df
        while 0 <= _rr < 8 and 0 <= _ff < 8:
            _b = _rr * 8 + _ff
            BETWEEN[_a * 64 + _b] = np.uint64(_acc)
            LINE[_a * 64 + _b] = (RAY[_d, _a] | RAY[(_d + 4) % 8, _a]
                                  | np.uint64(1 << _a))
            _acc |= 1 << _b
            _rr += _dr
            _ff += _df

DISTANCE = np.zeros(64 * 64, dtype=np.int64)
for _a in range(64):
    for _b in range(64):
        DISTANCE[_a * 64 + _b] = max(abs((_a >> 3) - (_b >> 3)),
                                     abs((_a & 7) - (_b & 7)))

CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 & ~CASTLE_WQ
CASTLE_MASK[7] = 15 & ~CASTLE_WK
CASTLE_MASK[4] = 15 & ~(CASTLE_WK | CASTLE_WQ)
CASTLE_MASK[56] = 15 & ~CASTLE_BQ
CASTLE_MASK[63] = 15 & ~CASTLE_BK
CASTLE_MASK[60] = 15 & ~(CASTLE_BK | CASTLE_BQ)

_rng = np.random.default_rng(0xC0FFEE)
Z_PIECE = _rng.integers(0, 1 << 63, size=(12, 64), dtype=np.uint64)
Z_SIDE = np.uint64(_rng.integers(0, 1 << 63, dtype=np.uint64))
Z_CASTLE = _rng.integers(0, 1 << 63, size=16, dtype=np.uint64)
Z_EP_FILE = _rng.integers(0, 1 << 63, size=8, dtype=np.uint64)


RELRANK = np.zeros((2, 64), dtype=np.int64)
for _s in range(64):
    RELRANK[0, _s] = _s >> 3
    RELRANK[1, _s] = 7 - (_s >> 3)


# ==========================================================================
# Bit helpers
# ==========================================================================
@njit(cache=False, inline="always")
def popcount(b):
    b = b - ((b >> U1) & K1)
    b = (b & K2) + ((b >> U2) & K2)
    b = (b + (b >> U4)) & K4
    return np.int64((b * KF) >> U56)


@njit(cache=False, inline="always")
def lsb(b):
    return INDEX64[((b & (~b + U1)) * DEBRUIJN) >> U58]


@njit(cache=False, inline="always")
def msb(b):
    b |= b >> U1
    b |= b >> U2
    b |= b >> U4
    b |= b >> U8
    b |= b >> U16
    b |= b >> U32
    return popcount(b) - 1


@njit(cache=False, inline="always")
def enc(fr, to, promo, flag):
    return fr | (to << 6) | (promo << 12) | (flag << 15)


@njit(cache=False, inline="always")
def m_from(m):
    return m & 63


@njit(cache=False, inline="always")
def m_to(m):
    return (m >> 6) & 63


@njit(cache=False, inline="always")
def m_promo(m):
    return (m >> 12) & 7


@njit(cache=False, inline="always")
def m_flag(m):
    return (m >> 15) & 7


@njit(cache=False, inline="always")
def is_capture_flag(f):
    return f == FLAG_CAPTURE or f == FLAG_EP or f == FLAG_PROMO_CAP


# ==========================================================================
# Sliding attacks
# ==========================================================================
@njit(cache=False, inline="always")
def ray_pos(sq, occ, d):
    att = RAY[d, sq]
    blk = att & occ
    if blk:
        att ^= RAY[d, lsb(blk)]
    return att


@njit(cache=False, inline="always")
def ray_neg(sq, occ, d):
    att = RAY[d, sq]
    blk = att & occ
    if blk:
        att ^= RAY[d, msb(blk)]
    return att


@njit(cache=False)
def bishop_attacks(sq, occ):
    return (ray_pos(sq, occ, 1) | ray_pos(sq, occ, 7)
            | ray_neg(sq, occ, 3) | ray_neg(sq, occ, 5))


@njit(cache=False)
def rook_attacks(sq, occ):
    return (ray_pos(sq, occ, 0) | ray_pos(sq, occ, 2)
            | ray_neg(sq, occ, 4) | ray_neg(sq, occ, 6))


@njit(cache=False)
def queen_attacks(sq, occ):
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


# ==========================================================================
# Attack queries
# ==========================================================================
@njit(cache=False)
def attackers_to(bb, sq, occ, by):
    them = bb[by]
    res = PAWN_ATT[1 - by, sq] & bb[PAWN + 1] & them
    res |= KNIGHT_ATT[sq] & bb[KNIGHT + 1] & them
    res |= KING_ATT[sq] & bb[KING + 1] & them
    bq = (bb[BISHOP + 1] | bb[QUEEN + 1]) & them
    if bq:
        res |= bishop_attacks(sq, occ) & bq
    rq = (bb[ROOK + 1] | bb[QUEEN + 1]) & them
    if rq:
        res |= rook_attacks(sq, occ) & rq
    return res & occ


@njit(cache=False)
def attackers_all(bb, sq, occ):
    res = PAWN_ATT[1, sq] & bb[PAWN + 1] & bb[0]
    res |= PAWN_ATT[0, sq] & bb[PAWN + 1] & bb[1]
    res |= KNIGHT_ATT[sq] & bb[KNIGHT + 1]
    res |= KING_ATT[sq] & bb[KING + 1]
    res |= bishop_attacks(sq, occ) & (bb[BISHOP + 1] | bb[QUEEN + 1])
    res |= rook_attacks(sq, occ) & (bb[ROOK + 1] | bb[QUEEN + 1])
    return res & occ


@njit(cache=False)
def square_attacked(bb, sq, occ, by):
    them = bb[by]
    if PAWN_ATT[1 - by, sq] & bb[PAWN + 1] & them:
        return True
    if KNIGHT_ATT[sq] & bb[KNIGHT + 1] & them:
        return True
    if KING_ATT[sq] & bb[KING + 1] & them:
        return True
    bq = (bb[BISHOP + 1] | bb[QUEEN + 1]) & them
    if bq and (bishop_attacks(sq, occ) & bq):
        return True
    rq = (bb[ROOK + 1] | bb[QUEEN + 1]) & them
    if rq and (rook_attacks(sq, occ) & rq):
        return True
    return False


@njit(cache=False)
def in_check(bb, side):
    return square_attacked(bb, lsb(bb[KING + 1] & bb[side]),
                           bb[0] | bb[1], 1 - side)


# ==========================================================================
# Legal move generation
# ==========================================================================
@njit(cache=False)
def gen_legal(bb, mb, side, castling, ep, moves, base, captures_only):
    """Fully legal generation. captures_only keeps captures and all
    promotions (including quiet ones), which is the quiescence move set."""
    us = side
    them = 1 - side
    own = bb[us]
    opp = bb[them]
    occ = own | opp
    n = base

    ksq = lsb(bb[KING + 1] & own)
    checkers = attackers_to(bb, ksq, occ, them)
    occ_nk = occ ^ SQ_BB[ksq]

    t = KING_ATT[ksq] & ~own
    if captures_only:
        t &= opp
    while t:
        to = lsb(t)
        t &= t - U1
        if not square_attacked(bb, to, occ_nk, them):
            moves[n] = enc(ksq, to, 0,
                           FLAG_CAPTURE if (opp & SQ_BB[to]) else FLAG_QUIET)
            n += 1

    nchk = popcount(checkers)
    if nchk > 1:
        return n

    if nchk == 1:
        tgt_all = checkers | BETWEEN[ksq * 64 + lsb(checkers)]
    else:
        tgt_all = FULL
        if not captures_only:
            if us == 0:
                if (castling & CASTLE_WK) and not (occ & np.uint64(0x60)):
                    if (not square_attacked(bb, 4, occ, 1)
                            and not square_attacked(bb, 5, occ, 1)
                            and not square_attacked(bb, 6, occ, 1)):
                        moves[n] = enc(4, 6, 0, FLAG_CASTLE)
                        n += 1
                if (castling & CASTLE_WQ) and not (occ & np.uint64(0x0E)):
                    if (not square_attacked(bb, 4, occ, 1)
                            and not square_attacked(bb, 3, occ, 1)
                            and not square_attacked(bb, 2, occ, 1)):
                        moves[n] = enc(4, 2, 0, FLAG_CASTLE)
                        n += 1
            else:
                if (castling & CASTLE_BK) and not (occ & np.uint64(0x6000000000000000)):
                    if (not square_attacked(bb, 60, occ, 0)
                            and not square_attacked(bb, 61, occ, 0)
                            and not square_attacked(bb, 62, occ, 0)):
                        moves[n] = enc(60, 62, 0, FLAG_CASTLE)
                        n += 1
                if (castling & CASTLE_BQ) and not (occ & np.uint64(0x0E00000000000000)):
                    if (not square_attacked(bb, 60, occ, 0)
                            and not square_attacked(bb, 59, occ, 0)
                            and not square_attacked(bb, 58, occ, 0)):
                        moves[n] = enc(60, 58, 0, FLAG_CASTLE)
                        n += 1

    tgt = tgt_all & opp if captures_only else tgt_all

    pinned = U0
    snipers = ((ROOK_PSEUDO[ksq] & (bb[ROOK + 1] | bb[QUEEN + 1]))
               | (BISHOP_PSEUDO[ksq] & (bb[BISHOP + 1] | bb[QUEEN + 1]))) & opp
    s = snipers
    while s:
        sq = lsb(s)
        s &= s - U1
        b = BETWEEN[ksq * 64 + sq] & occ
        if b and (b & (b - U1)) == U0 and (b & own):
            pinned |= b

    t = bb[KNIGHT + 1] & own & ~pinned
    while t:
        fr = lsb(t)
        t &= t - U1
        d = KNIGHT_ATT[fr] & ~own & tgt
        while d:
            to = lsb(d)
            d &= d - U1
            moves[n] = enc(fr, to, 0,
                           FLAG_CAPTURE if (opp & SQ_BB[to]) else FLAG_QUIET)
            n += 1

    for pt in (BISHOP, ROOK, QUEEN):
        t = bb[pt + 1] & own
        while t:
            fr = lsb(t)
            t &= t - U1
            if pt == BISHOP:
                d = bishop_attacks(fr, occ)
            elif pt == ROOK:
                d = rook_attacks(fr, occ)
            else:
                d = queen_attacks(fr, occ)
            d &= ~own & tgt
            if pinned & SQ_BB[fr]:
                d &= LINE[ksq * 64 + fr]
            while d:
                to = lsb(d)
                d &= d - U1
                moves[n] = enc(fr, to, 0,
                               FLAG_CAPTURE if (opp & SQ_BB[to]) else FLAG_QUIET)
                n += 1

    t = bb[PAWN + 1] & own
    while t:
        fr = lsb(t)
        t &= t - U1
        allowed = FULL
        if pinned & SQ_BB[fr]:
            allowed = LINE[ksq * 64 + fr]
        rel = RELRANK[us, fr]

        if rel < 7:
            to = fr + 8 if us == 0 else fr - 8
            if mb[to] == 0:
                if SQ_BB[to] & tgt_all & allowed:
                    if rel == 6:
                        moves[n] = enc(fr, to, QUEEN, FLAG_PROMO)
                        moves[n + 1] = enc(fr, to, ROOK, FLAG_PROMO)
                        moves[n + 2] = enc(fr, to, BISHOP, FLAG_PROMO)
                        moves[n + 3] = enc(fr, to, KNIGHT, FLAG_PROMO)
                        n += 4
                    elif not captures_only:
                        moves[n] = enc(fr, to, 0, FLAG_QUIET)
                        n += 1
                if rel == 1 and not captures_only:
                    to2 = fr + 16 if us == 0 else fr - 16
                    if mb[to2] == 0 and (SQ_BB[to2] & tgt_all & allowed):
                        moves[n] = enc(fr, to2, 0, FLAG_DOUBLE)
                        n += 1

        c = PAWN_ATT[us, fr] & opp & tgt_all & allowed
        while c:
            to = lsb(c)
            c &= c - U1
            if rel == 6:
                moves[n] = enc(fr, to, QUEEN, FLAG_PROMO_CAP)
                moves[n + 1] = enc(fr, to, ROOK, FLAG_PROMO_CAP)
                moves[n + 2] = enc(fr, to, BISHOP, FLAG_PROMO_CAP)
                moves[n + 3] = enc(fr, to, KNIGHT, FLAG_PROMO_CAP)
                n += 4
            else:
                moves[n] = enc(fr, to, 0, FLAG_CAPTURE)
                n += 1

        if ep >= 0 and (PAWN_ATT[us, fr] & SQ_BB[ep]):
            capsq = ep - 8 if us == 0 else ep + 8
            if (SQ_BB[ep] & tgt_all) or (SQ_BB[capsq] & checkers):
                occ2 = (occ ^ SQ_BB[fr] ^ SQ_BB[capsq]) | SQ_BB[ep]
                bq = (bb[BISHOP + 1] | bb[QUEEN + 1]) & opp
                rq = (bb[ROOK + 1] | bb[QUEEN + 1]) & opp
                ok = True
                if bq and (bishop_attacks(ksq, occ2) & bq):
                    ok = False
                if ok and rq and (rook_attacks(ksq, occ2) & rq):
                    ok = False
                if ok:
                    moves[n] = enc(fr, ep, 0, FLAG_EP)
                    n += 1
    return n


# ==========================================================================
# Make / unmake
# ==========================================================================
@njit(cache=False)
def make_move(bb, mb, side, castling, ep, halfmove, key, move, undo, ply):
    fr = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    us = side
    them = 1 - side

    new_key = key ^ Z_SIDE
    if ep >= 0:
        new_key ^= Z_EP_FILE[ep & 7]
    new_key ^= Z_CASTLE[castling]

    pt = mb[fr]
    frbb = SQ_BB[fr]
    tobb = SQ_BB[to]

    bb[us] ^= frbb
    bb[pt + 1] ^= frbb
    mb[fr] = 0
    new_key ^= Z_PIECE[us * 6 + pt - 1, fr]

    cap = 0
    new_ep = -1
    new_half = halfmove + 1

    if flag == FLAG_EP:
        capsq = to - 8 if us == 0 else to + 8
        cbb = SQ_BB[capsq]
        bb[them] ^= cbb
        bb[PAWN + 1] ^= cbb
        mb[capsq] = 0
        new_key ^= Z_PIECE[them * 6 + PAWN - 1, capsq]
        new_half = 0
    elif flag == FLAG_CAPTURE or flag == FLAG_PROMO_CAP:
        cap = mb[to]
        bb[them] ^= tobb
        bb[cap + 1] ^= tobb
        new_key ^= Z_PIECE[them * 6 + cap - 1, to]
        new_half = 0

    newpt = promo if promo else pt
    bb[us] |= tobb
    bb[newpt + 1] |= tobb
    mb[to] = newpt
    new_key ^= Z_PIECE[us * 6 + newpt - 1, to]

    if flag == FLAG_CASTLE:
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        rbb = SQ_BB[rf] | SQ_BB[rt]
        bb[us] ^= rbb
        bb[ROOK + 1] ^= rbb
        mb[rf] = 0
        mb[rt] = ROOK
        new_key ^= Z_PIECE[us * 6 + ROOK - 1, rf]
        new_key ^= Z_PIECE[us * 6 + ROOK - 1, rt]
    elif flag == FLAG_DOUBLE:
        new_ep = (fr + to) // 2
        new_half = 0
    elif pt == PAWN:
        new_half = 0

    new_castling = castling & CASTLE_MASK[fr] & CASTLE_MASK[to]
    new_key ^= Z_CASTLE[new_castling]
    if new_ep >= 0:
        new_key ^= Z_EP_FILE[new_ep & 7]

    undo[ply] = cap
    return new_castling, new_ep, new_half, new_key


@njit(cache=False)
def unmake_move(bb, mb, side, move, undo, ply):
    fr = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    us = side
    them = 1 - side

    frbb = SQ_BB[fr]
    tobb = SQ_BB[to]
    newpt = mb[to]

    bb[us] ^= tobb
    bb[newpt + 1] ^= tobb
    mb[to] = 0

    origpt = PAWN if promo else newpt
    bb[us] |= frbb
    bb[origpt + 1] |= frbb
    mb[fr] = origpt

    if flag == FLAG_EP:
        capsq = to - 8 if us == 0 else to + 8
        cbb = SQ_BB[capsq]
        bb[them] |= cbb
        bb[PAWN + 1] |= cbb
        mb[capsq] = PAWN
    elif flag == FLAG_CAPTURE or flag == FLAG_PROMO_CAP:
        cap = undo[ply]
        bb[them] |= tobb
        bb[cap + 1] |= tobb
        mb[to] = cap
    elif flag == FLAG_CASTLE:
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        rbb = SQ_BB[rf] | SQ_BB[rt]
        bb[us] ^= rbb
        bb[ROOK + 1] ^= rbb
        mb[rt] = 0
        mb[rf] = ROOK


@njit(cache=False)
def compute_key(bb, mb, side, castling, ep):
    k = U0
    for sq in range(64):
        pt = mb[sq]
        if pt:
            c = 0 if (bb[0] & SQ_BB[sq]) else 1
            k ^= Z_PIECE[c * 6 + pt - 1, sq]
    if side == 1:
        k ^= Z_SIDE
    k ^= Z_CASTLE[castling]
    if ep >= 0:
        k ^= Z_EP_FILE[ep & 7]
    return k


# ==========================================================================
# Python boundary
# ==========================================================================
def fen_to_state(fen: str):
    parts = fen.split()
    mb = np.zeros(64, dtype=np.int8)
    bb = np.zeros(8, dtype=np.uint64)
    pmap = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}
    rank, file_ = 7, 0
    for ch in parts[0]:
        if ch == "/":
            rank -= 1
            file_ = 0
        elif ch.isdigit():
            file_ += int(ch)
        else:
            sq = rank * 8 + file_
            pt = pmap[ch.lower()]
            col = 0 if ch.isupper() else 1
            mb[sq] = pt
            bb[col] |= np.uint64(1 << sq)
            bb[pt + 1] |= np.uint64(1 << sq)
            file_ += 1
    side = 0 if parts[1] == "w" else 1
    castling = 0
    if len(parts) > 2:
        if "K" in parts[2]:
            castling |= CASTLE_WK
        if "Q" in parts[2]:
            castling |= CASTLE_WQ
        if "k" in parts[2]:
            castling |= CASTLE_BK
        if "q" in parts[2]:
            castling |= CASTLE_BQ
    ep = -1
    if len(parts) > 3 and parts[3] != "-":
        ep = (ord(parts[3][0]) - 97) + (int(parts[3][1]) - 1) * 8
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    return bb, mb, side, castling, ep, halfmove


_PROMO_CHAR = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


def move_to_uci(move: int) -> str:
    s = chess.square_name(move & 63) + chess.square_name((move >> 6) & 63)
    promo = (move >> 12) & 7
    if promo:
        s += _PROMO_CHAR[promo]
    return s
