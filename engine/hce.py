"""Hand-written evaluation: material, piece-square tables, mobility, pawn
structure, king safety, passed pawns, space. This is the baseline every
network was compared against, and the fallback if no network loads.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .config import *  # noqa: F401,F403
from .board import *  # noqa: F401,F403

# ==========================================================================
# Evaluation tables (ported verbatim from v10)
# ==========================================================================
PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE = 100, 320, 330
ROOK_VALUE, QUEEN_VALUE = 500, 900
PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 10000], dtype=np.int64)

_PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]
_QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
_KING_MG_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]
_KING_EG_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

# PST[colour, piece_type, square]
PST = np.zeros((2, 7, 64), dtype=np.int64)
for _pt, _tbl in ((PAWN, _PAWN_PST), (KNIGHT, _KNIGHT_PST), (BISHOP, _BISHOP_PST),
                  (ROOK, _ROOK_PST), (QUEEN, _QUEEN_PST)):
    for _s in range(64):
        PST[0, _pt, _s] = _tbl[_s ^ 56]
        PST[1, _pt, _s] = _tbl[_s]
KING_PST_MG = np.zeros((2, 64), dtype=np.int64)
KING_PST_EG = np.zeros((2, 64), dtype=np.int64)
for _s in range(64):
    KING_PST_MG[0, _s] = _KING_MG_PST[_s ^ 56]
    KING_PST_MG[1, _s] = _KING_MG_PST[_s]
    KING_PST_EG[0, _s] = _KING_EG_PST[_s ^ 56]
    KING_PST_EG[1, _s] = _KING_EG_PST[_s]


PASSED_MASK = np.zeros((2, 64), dtype=np.uint64)
KING_SHIELD = np.zeros((2, 64), dtype=np.uint64)
for _c in (0, 1):
    for _s in range(64):
        _r, _f = divmod(_s, 8)
        _files = FILE_BB[_f] | ADJ_FILE[_f]
        _ahead = U0
        if _c == 0:
            for _o in range(_r + 1, 8):
                _ahead |= RANK_BB[_o]
        else:
            for _o in range(_r):
                _ahead |= RANK_BB[_o]
        PASSED_MASK[_c, _s] = _files & _ahead
        _ranks = U0
        for _o in ((_r + 1, _r + 2) if _c == 0 else (_r - 1, _r - 2)):
            if 0 <= _o <= 7:
                _ranks |= RANK_BB[_o]
        KING_SHIELD[_c, _s] = _files & _ranks

MAX_PHASE = 24
# The base is subtracted from every count, so with equal material it cancels
# in the difference and only its distance from the true average matters. The
# mobility area removes one or two squares from a typical piece, so each base
# drops by about that much; leaving them alone would quietly devalue every
# piece by a few centipawns of material.
KNIGHT_MOBILITY = 4
KNIGHT_MOBILITY_BASE = 3 if USE_MOBILITY_AREA else 4
BISHOP_MOBILITY = 4
BISHOP_MOBILITY_BASE = 5 if USE_MOBILITY_AREA else 6
ROOK_MOBILITY = 2
ROOK_MOBILITY_BASE = 6 if USE_MOBILITY_AREA else 7
QUEEN_MOBILITY = 1
QUEEN_MOBILITY_BASE = 12 if USE_MOBILITY_AREA else 14
BISHOP_PAIR = 30
# Light squares, for the opposite-coloured-bishop test. a1 is square 0 and is
# dark, so a square is light when rank + file is odd.
LIGHT_BB = np.uint64(0)
for _s in range(64):
    if (((_s >> 3) + (_s & 7)) & 1) == 1:
        LIGHT_BB |= np.uint64(1 << _s)
# A pure opposite-bishop ending -- one bishop each, on opposite colours, no
# knights, rooks or queens left -- is drawish out of all proportion to the
# material on the board. The scale is a fraction out of OCB_SCALE_DEN applied
# to the whole score, so a dead-drawn B+2P v B+1P stops reading +116.
OCB_SCALE_DEN = 128
OCB_BASE_SCALE = 40     # equal pawns: 0.31x
OCB_PAWN_SCALE = 12     # per pawn of material advantage, +0.09x
ISOLATED_PAWN = -14
DOUBLED_PAWN = -18
PROTECTED_PASSER = 15
PASSED_BONUS = np.array([0, 5, 12, 24, 45, 75, 120, 0], dtype=np.int64)
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN_FILE = 10
ROOK_ON_SEVENTH = 35
SHIELD_PAWN = 14
KING_OPEN_FILE = -24
KING_ACTIVITY = 15
TEMPO = 12

# ---- batch 2: king safety -------------------------------------------------
# The zone is the king, its eight neighbours, and the same shifted one rank
# toward the enemy: an attack is built on the squares in front of the king,
# not the ones behind it.
KING_ZONE = np.zeros((2, 64), dtype=np.uint64)
for _c in range(2):
    for _s in range(64):
        _z = int(KING_ATT[_s]) | (1 << _s)
        _z |= (_z << 8) if _c == 0 else (_z >> 8)
        KING_ZONE[_c, _s] = np.uint64(_z & 0xFFFFFFFFFFFFFFFF)

# Units per zone square hit, indexed by piece type. A knight is worth as much
# as a bishop here, which is the entire point: numbav1 counted it as zero.
ATT_WEIGHT = np.array([0, 1, 2, 2, 3, 5, 0], dtype=np.int64)
# Quadratic, so four attackers cost far more than four times one attacker.
SAFETY_TABLE = np.zeros(80, dtype=np.int64)
for _i in range(80):
    SAFETY_TABLE[_i] = min(650, (_i * _i) // 3)
# A rook or queen on an open or half-open file beside the king. This is what
# 29.h4 in round 56 handed Black, scored at cp18 at the time.
ROOK_ON_KING_FILE_UNITS = 4
# Enemy pawns advancing on the king, by their relative rank.
STORM_UNITS = np.array([0, 0, 0, 1, 2, 4, 6, 0], dtype=np.int64)

# ---- batch 2: passed pawns ------------------------------------------------
# Percent scaling applied to PASSED_BONUS at each end of the phase range.
PASSER_MG_SCALE = 100
PASSER_EG_SCALE = 195
CONNECTED_PASSER = np.array([0, 0, 4, 10, 20, 34, 52, 0], dtype=np.int64)
PASSER_BLOCKED = np.array([0, 0, -3, -8, -16, -28, -44, 0], dtype=np.int64)
PASSER_STOP_HIT = np.array([0, 0, -2, -6, -12, -22, -34, 0], dtype=np.int64)
ROOK_BEHIND_PASSER = 18
PASSER_KING_DIST = 4

# ---- v6: space ------------------------------------------------------------
# The zone is relative ranks four to six on files b to g: the part of the board
# a middlegame is fought over. A square counts as gained when our own pawns
# hold it or attack it and no enemy pawn covers it, which is the definition
# that makes a pawn advance pay and a piece shuffle cost nothing.
SPACE_ZONE = np.zeros(2, dtype=np.uint64)
for _c in (0, 1):
    for _s in range(64):
        if 3 <= RELRANK[_c, _s] <= 5 and 1 <= (_s & 7) <= 6:
            SPACE_ZONE[_c] |= SQ_BB[_s]
# Centipawns per gained square, faded out with the phase. Eighteen squares is
# the ceiling, so this is bounded at SPACE_WEIGHT * 18 and in practice runs at
# four to ten squares a side. First constant to tune if v6 plays loosely.
SPACE_WEIGHT = 4
SPACE_MIN_PHASE = 10

# ---- v6: pawn threats -----------------------------------------------------
# Indexed by the piece type an enemy pawn is attacking. Deliberately below the
# usual engine values: at depth thirteen an overpriced threat term makes the
# engine flinch at pawn advances it could simply meet.
PAWN_THREAT = np.array([0, 0, -35, -35, -48, -60, 0], dtype=np.int64)

# ---- v6: tapered pawn table -----------------------------------------------
# The middlegame table is numbav5's and is unchanged. This is the endgame half:
# flat by rank, because once the pieces come off the only thing a pawn wants is
# to be further up the board. Kept modest so it does not double-count with
# PASSED_BONUS, which already scales to 195 percent in the endgame.
_PAWN_PST_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     90,  90,  90,  90,  90,  90,  90,  90,
     50,  50,  50,  50,  50,  50,  50,  50,
     28,  28,  28,  28,  28,  28,  28,  28,
     15,  15,  15,  15,  15,  15,  15,  15,
      8,   8,   8,   8,   8,   8,   8,   8,
      4,   4,   4,   4,   4,   4,   4,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PAWN_PST_EG = np.zeros((2, 64), dtype=np.int64)
for _s in range(64):
    PAWN_PST_EG[0, _s] = _PAWN_PST_EG[_s ^ 56]
    PAWN_PST_EG[1, _s] = _PAWN_PST_EG[_s]


# ==========================================================================
# Evaluation (port of v10 _colour_score, term for term and weight for weight)
# ==========================================================================
@njit(cache=False)
def colour_score(bb, mb, colour, phase):
    own = bb[colour]
    opp = bb[1 - colour]
    occ = own | opp
    own_pawns = bb[PAWN + 1] & own
    enemy_pawns = bb[PAWN + 1] & opp
    all_pawns = own_pawns | enemy_pawns

    # Every square an enemy pawn covers. Three of the four v6 terms read it,
    # so it is built once, from the same PAWN_ATT table the passer code uses
    # rather than from shifts, which is one place a uint64 wrap is easy to get
    # wrong and impossible to see afterwards.
    opp_pawn_att = U0
    if USE_MOBILITY_AREA or USE_SPACE or USE_PAWN_THREATS:
        t = enemy_pawns
        while t:
            sq = lsb(t)
            t &= t - U1
            opp_pawn_att |= PAWN_ATT[1 - colour, sq]

    own_pawn_att = U0
    if USE_SPACE:
        t = own_pawns
        while t:
            sq = lsb(t)
            t &= t - U1
            own_pawn_att |= PAWN_ATT[colour, sq]

    # A square an enemy pawn attacks is not somewhere a knight, bishop, rook
    # or queen can stand, so counting it as mobility is counting a square the
    # piece does not have. This is what let numbav5 score the knight it had
    # parked on b8 in game 80 as though it were doing something.
    if USE_MOBILITY_AREA:
        available = ~(own | opp_pawn_att)
    else:
        available = ~own
    score = 0

    passers = U0
    t = own_pawns
    while t:
        sq = lsb(t)
        t &= t - U1
        if USE_PAWN_PST_TAPER:
            score += PAWN_VALUE + ((PST[colour, PAWN, sq] * phase
                                    + PAWN_PST_EG[colour, sq] * (MAX_PHASE - phase))
                                   // MAX_PHASE)
        else:
            score += PAWN_VALUE + PST[colour, PAWN, sq]
        if not (ADJ_FILE[sq & 7] & own_pawns):
            score += ISOLATED_PAWN
        if not (PASSED_MASK[colour, sq] & enemy_pawns):
            passers |= SQ_BB[sq]
            if not USE_PASSER_V2:
                score += PASSED_BONUS[RELRANK[colour, sq]]
                if PAWN_ATT[1 - colour, sq] & own_pawns:
                    score += PROTECTED_PASSER

    if USE_PASSER_V2 and passers:
        own_rooks = bb[ROOK + 1] & own
        own_king = lsb(bb[KING + 1] & own)
        opp_king = lsb(bb[KING + 1] & opp)
        t = passers
        while t:
            sq = lsb(t)
            t &= t - U1
            r = RELRANK[colour, sq]
            # A passer is worth roughly twice as much once the pieces are off.
            # numbav1 paid the same rank bonus in both, which is how round 57
            # scored three connected passers at half a pawn.
            scale = (PASSER_MG_SCALE * phase
                     + PASSER_EG_SCALE * (MAX_PHASE - phase)) // MAX_PHASE
            bonus = PASSED_BONUS[r] * scale // 100

            if PAWN_ATT[1 - colour, sq] & own_pawns:
                bonus += PROTECTED_PASSER
            # Two passers side by side defend each other's stop squares, which
            # is worth far more than either alone.
            if ADJ_FILE[sq & 7] & passers:
                bonus += CONNECTED_PASSER[r]

            stop = sq + 8 if colour == 0 else sq - 8
            if 0 <= stop < 64:
                if SQ_BB[stop] & occ:
                    bonus += PASSER_BLOCKED[r]
                elif square_attacked(bb, stop, occ, 1 - colour):
                    bonus += PASSER_STOP_HIT[r]
                if phase < MAX_PHASE:
                    # Whose king gets there first only matters as the board
                    # empties, so this fades in with the endgame.
                    d_own = DISTANCE[own_king * 64 + stop]
                    d_opp = DISTANCE[opp_king * 64 + stop]
                    bonus += (PASSER_KING_DIST * (d_opp - d_own) * r
                              * (MAX_PHASE - phase) // MAX_PHASE) // 4

            if own_rooks & FILE_BB[sq & 7]:
                behind = FILE_BB[sq & 7] & own_rooks
                rsq = lsb(behind)
                if (RELRANK[colour, rsq] < r):
                    bonus += ROOK_BEHIND_PASSER

            score += bonus

    for f in range(8):
        stacked = popcount(own_pawns & FILE_BB[f])
        if stacked > 1:
            score += DOUBLED_PAWN * (stacked - 1)

    t = bb[KNIGHT + 1] & own
    while t:
        sq = lsb(t)
        t &= t - U1
        mob = popcount(KNIGHT_ATT[sq] & available)
        score += (KNIGHT_VALUE + PST[colour, KNIGHT, sq]
                  + (mob - KNIGHT_MOBILITY_BASE) * KNIGHT_MOBILITY)

    bishops = bb[BISHOP + 1] & own
    if popcount(bishops) >= 2:
        score += BISHOP_PAIR
    t = bishops
    while t:
        sq = lsb(t)
        t &= t - U1
        mob = popcount(bishop_attacks(sq, occ) & available)
        score += (BISHOP_VALUE + PST[colour, BISHOP, sq]
                  + (mob - BISHOP_MOBILITY_BASE) * BISHOP_MOBILITY)

    t = bb[ROOK + 1] & own
    while t:
        sq = lsb(t)
        t &= t - U1
        mob = popcount(rook_attacks(sq, occ) & available)
        score += (ROOK_VALUE + PST[colour, ROOK, sq]
                  + (mob - ROOK_MOBILITY_BASE) * ROOK_MOBILITY)
        fbb = FILE_BB[sq & 7]
        if not (fbb & all_pawns):
            score += ROOK_OPEN_FILE
        elif not (fbb & own_pawns):
            score += ROOK_SEMI_OPEN_FILE
        if RELRANK[colour, sq] == 6:
            score += ROOK_ON_SEVENTH

    t = bb[QUEEN + 1] & own
    while t:
        sq = lsb(t)
        t &= t - U1
        mob = popcount(queen_attacks(sq, occ) & available)
        score += (QUEEN_VALUE + PST[colour, QUEEN, sq]
                  + (mob - QUEEN_MOBILITY_BASE) * QUEEN_MOBILITY)

    # Space. Counted from the pawns alone: a piece standing on a square is not
    # space, it is a piece, and it can be chased off. Pawns cannot.
    if USE_SPACE and phase >= SPACE_MIN_PHASE:
        gained = SPACE_ZONE[colour] & (own_pawn_att | own_pawns) & ~opp_pawn_att
        score += SPACE_WEIGHT * popcount(gained) * phase // MAX_PHASE

    # A pawn attacking a piece. Charged whether or not the piece is defended,
    # because the pawn is cheaper than anything it can be attacking and the
    # piece has to move; that is the whole cost of the tempo White spent forty
    # moves collecting in game 80 with a4, b4, a5, d4, d5, c4, h4 and h5.
    if USE_PAWN_THREATS:
        t = opp_pawn_att & own & ~bb[PAWN + 1]
        while t:
            sq = lsb(t)
            t &= t - U1
            score += PAWN_THREAT[mb[sq]]

    ksq = lsb(bb[KING + 1] & own)
    score += ((KING_PST_MG[colour, ksq] * phase
               + KING_PST_EG[colour, ksq] * (MAX_PHASE - phase)) // MAX_PHASE)

    if phase > 8:
        shield = popcount(own_pawns & KING_SHIELD[colour, ksq])
        if shield > 3:
            shield = 3
        score += (shield - 3) * SHIELD_PAWN * phase // MAX_PHASE
        kf = ksq & 7
        exposed = 0
        for f in range(kf - 1, kf + 2):
            if 0 <= f <= 7 and not (FILE_BB[f] & own_pawns):
                exposed += 1
        score += exposed * KING_OPEN_FILE * phase // MAX_PHASE
    else:
        if enemy_pawns:
            best = 99
            t = enemy_pawns
            while t:
                sq = lsb(t)
                t &= t - U1
                d = DISTANCE[ksq * 64 + sq]
                if d < best:
                    best = d
            score += KING_ACTIVITY * (7 - best) * (MAX_PHASE - phase) // MAX_PHASE

    if USE_KING_SAFETY_V2:
        zone = KING_ZONE[colour, ksq]
        units = 0
        attackers = 0

        t = bb[KNIGHT + 1] & opp
        while t:
            sq = lsb(t)
            t &= t - U1
            hits = popcount(KNIGHT_ATT[sq] & zone)
            if hits:
                attackers += 1
                units += ATT_WEIGHT[KNIGHT] * hits

        t = bb[BISHOP + 1] & opp
        while t:
            sq = lsb(t)
            t &= t - U1
            hits = popcount(bishop_attacks(sq, occ) & zone)
            if hits:
                attackers += 1
                units += ATT_WEIGHT[BISHOP] * hits

        t = bb[ROOK + 1] & opp
        while t:
            sq = lsb(t)
            t &= t - U1
            hits = popcount(rook_attacks(sq, occ) & zone)
            if hits:
                attackers += 1
                units += ATT_WEIGHT[ROOK] * hits
            # A rook bearing down a file the king has no pawn on. This is what
            # 29.h4 conceded in round 56, and numbav1 charged twenty-four
            # centipawns for it.
            if abs((sq & 7) - (ksq & 7)) <= 1 and not (FILE_BB[sq & 7] & own_pawns):
                units += ROOK_ON_KING_FILE_UNITS

        t = bb[QUEEN + 1] & opp
        while t:
            sq = lsb(t)
            t &= t - U1
            hits = popcount(queen_attacks(sq, occ) & zone)
            if hits:
                attackers += 1
                units += ATT_WEIGHT[QUEEN] * hits
            if abs((sq & 7) - (ksq & 7)) <= 1 and not (FILE_BB[sq & 7] & own_pawns):
                units += ROOK_ON_KING_FILE_UNITS

        # Pawns, which numbav1 ignored entirely -- both as zone attackers and
        # as a storm rolling up the board.
        pawn_hits = 0
        kf = ksq & 7
        t = enemy_pawns
        while t:
            sq = lsb(t)
            t &= t - U1
            if PAWN_ATT[1 - colour, sq] & zone:
                pawn_hits += 1
            if abs((sq & 7) - kf) <= 1:
                units += STORM_UNITS[RELRANK[1 - colour, sq]]
        if pawn_hits:
            units += ATT_WEIGHT[PAWN] * pawn_hits

        if units > 79:
            units = 79
        penalty = SAFETY_TABLE[units]
        if attackers < 2:
            # One piece pointing at a king is not an attack.
            penalty //= 3
        score -= penalty * phase // MAX_PHASE
    else:
        king_zone = KING_ATT[ksq] | SQ_BB[ksq]
        attacked = U0
        t = opp & (bb[QUEEN + 1] | bb[ROOK + 1] | bb[BISHOP + 1])
        while t:
            sq = lsb(t)
            t &= t - U1
            pt = mb[sq]
            if pt == BISHOP:
                attacked |= bishop_attacks(sq, occ)
            elif pt == ROOK:
                attacked |= rook_attacks(sq, occ)
            else:
                attacked |= queen_attacks(sq, occ)
        penalty = -12 * popcount(king_zone & attacked)
        t = bb[QUEEN + 1] & opp
        while t:
            sq = lsb(t)
            t &= t - U1
            d = DISTANCE[ksq * 64 + sq]
            if d <= 3:
                penalty -= 15 * (4 - d)
        score += penalty * phase // MAX_PHASE

    return score


@njit(cache=False)
def ocb_scale_num(bb):
    """Numerator out of OCB_SCALE_DEN for a pure opposite-bishop ending.

    Returns OCB_SCALE_DEN (no scaling) unless the position is bishops-and-
    pawns only, one bishop a side, and the two bishops stand on squares of
    different colours. Round 85 traded the last rooks into exactly this at a
    reported +147, then reported +116 for fifty-five straight moves while
    shuffling, and drew on insufficient material. The material was real; the
    winning chances were not.
    """
    if bb[KNIGHT + 1] or bb[ROOK + 1] or bb[QUEEN + 1]:
        return OCB_SCALE_DEN
    wb = bb[BISHOP + 1] & bb[0]
    sb = bb[BISHOP + 1] & bb[1]
    if popcount(wb) != 1 or popcount(sb) != 1:
        return OCB_SCALE_DEN
    # Same colour complex is not drawish, so leave it alone.
    if ((wb & LIGHT_BB) != U0) == ((sb & LIGHT_BB) != U0):
        return OCB_SCALE_DEN
    d = popcount(bb[PAWN + 1] & bb[0]) - popcount(bb[PAWN + 1] & bb[1])
    if d < 0:
        d = -d
    n = OCB_BASE_SCALE + OCB_PAWN_SCALE * d
    if n > OCB_SCALE_DEN:
        n = OCB_SCALE_DEN
    return n
