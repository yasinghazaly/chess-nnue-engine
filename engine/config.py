"""Feature flags, constants and the layout of the search buffers.

Numba treats module-level globals as compile-time constants, so every flag
here is folded into the compiled code of whichever module reads it.
"""
from __future__ import annotations

import numpy as np

# ==========================================================================
# Feature flags
#
# Numba treats a module-level global as a compile-time constant, so each of
# these folds away entirely: a False flag emits no branch and no code. Flip
# one, re-import, and the engine is the older engine in that respect. This is
# the bisection handle for the batch.
# ==========================================================================
USE_MATE_DEPTH_GUARD = True      # tier 0: do not stop on an unverified mate
USE_FORCED_MOVE_REPFIX = True    # tier 0: record forced moves for threefold
USE_ASPIRATION_TIME_GUARD = True # tier 0: soft limit inside the window loop

USE_RFP = True                   # tier 1: reverse futility / static null move
USE_LMP = True                   # tier 1: late move (count) pruning
USE_LMR_TABLE = True             # tier 1: logarithmic reduction table
USE_SEE_PRUNE_CAPTURE = True     # tier 1: SEE pruning, captures (now free)
USE_SEE_PRUNE_QUIET = True       # tier 1: SEE pruning, quiets (now filtered)
USE_QS_TT = True                 # tier 1: transposition probe in quiescence
USE_NULL_ADAPTIVE = True         # tier 1: eval-scaled null-move reduction
BIG_TABLES = True                # tier 1: 22-bit TT, 21-bit eval cache
SAFE_CLOCK = True                # v3: wider overhead, tighter hard cap
USE_KING_SAFETY_V2 = True        # batch 2: attack-unit king safety
USE_PASSER_V2 = True             # batch 2: phase-scaled, connected passers
ADAPTIVE_INCREMENT = True        # v3: measure the increment, do not assume it

# ==========================================================================
# ACTIVITY PACK -- the whole of numbav6. Evaluation only; the search is
# byte-for-byte numbav5. Each flag is independent, so a bad arena result can
# be bisected by flipping one boolean rather than by rebuilding the file.
# ==========================================================================
USE_MOBILITY_AREA = True    # v6: mobility excludes squares enemy pawns cover
USE_SPACE = True            # v6: space gained in the middle of the board
USE_PAWN_THREATS = True     # v6: an enemy pawn attacking one of our pieces
USE_PAWN_PST_TAPER = True   # v6: pawns want to advance once the pieces are off

# ==========================================================================
# v7 -- two changes, one in the root loop and one in the evaluation. Both
# come from a specific rated game; see the header.
# ==========================================================================
USE_OCB_SCALE = True        # v7: scale a pure opposite-bishop ending to a draw

# ==========================================================================
# v8 -- the learned evaluation. Search untouched; this replaces colour_score
# and nothing else. Set USE_NNUE False and the engine is numbav7 exactly.
# ==========================================================================
USE_NNUE = True             # v8: 769 -> 512 -> 1 network instead of the HCE

# ==========================================================================
# v9/v10 -- the bucketed learned evaluation. When it loads it takes precedence
# over the v8 network. Set USE_NNUE_V2 False and the engine is numbav8.
# v9 ran the second network (4 buckets, 256); v10, this engine, runs the
# third network in the same form (8 buckets, 384), so only constants change.
# ==========================================================================
USE_NNUE_V2 = True          # v10: 2 x (6144 -> 384) 8 king buckets, SCReLU, 8 heads (NN3)

# ==========================================================================
# Constants
# ==========================================================================
U0 = np.uint64(0)
U1 = np.uint64(1)
U2 = np.uint64(2)
U4 = np.uint64(4)
U8 = np.uint64(8)
U16 = np.uint64(16)
U32 = np.uint64(32)
U56 = np.uint64(56)
U58 = np.uint64(58)
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)

K1 = np.uint64(0x5555555555555555)
K2 = np.uint64(0x3333333333333333)
K4 = np.uint64(0x0F0F0F0F0F0F0F0F)
KF = np.uint64(0x0101010101010101)
DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

# Mutually exclusive move flags. The previous engine used OR-able bit values,
# so FLAG_EP | FLAG_CAPTURE collided with FLAG_CASTLE and every castle ran the
# en-passant branch of make_move.
FLAG_QUIET = 0
FLAG_CAPTURE = 1
FLAG_EP = 2
FLAG_CASTLE = 3
FLAG_DOUBLE = 4
FLAG_PROMO = 5
FLAG_PROMO_CAP = 6

NO_MOVE = 0
MAX_MOVES = 220
MAX_PLY = 128
SEARCH_PLY_LIMIT = 100
MAX_QDEPTH = 6

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

MATE_SCORE = 1_000_000
MATE_BOUND = MATE_SCORE - 1000
INFINITY = 10_000_000


# ==========================================================================
# Mutable search state
#
# Numba freezes module-level arrays as READ-ONLY, so nothing the search
# mutates can live in a global. Everything is packed into four typed buffers
# that are threaded through the search as arguments. Packing rather than
# passing twenty separate arrays keeps the recursive call cheap and the
# offsets below are compile-time constants, so indexing costs nothing.
# ==========================================================================
# 22/21 rather than numbav2's 23/22. Peak RSS went 407 -> 559 MB in v2, and
# the arena runs more than one process on a box, so the extra 80 MB bought
# little and was a plausible contributor to the three crashes.
TT_BITS = 22 if BIG_TABLES else 21
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)
EV_BITS = 21 if BIG_TABLES else 20
EV_SIZE = 1 << EV_BITS
EV_MASK = np.uint64(EV_SIZE - 1)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# u64 buffer: TT keys | eval-cache keys | search path | game repetition keys
EV_KEY_OFF = TT_SIZE
PATH_OFF = TT_SIZE + EV_SIZE
REP_OFF = PATH_OFF + MAX_PLY
REP_CAP = 256
U64_LEN = REP_OFF + REP_CAP

# i32 buffer: TT scores | TT moves | eval values | move lists | move scores
#             | history | countermoves | killers
TT_MOVE_OFF = TT_SIZE
EV_VAL_OFF = 2 * TT_SIZE
MOVES_OFF = 2 * TT_SIZE + EV_SIZE
MSCORE_DELTA = MAX_PLY * MAX_MOVES
MSCORE_OFF = MOVES_OFF + MSCORE_DELTA
HIST_OFF = MSCORE_OFF + MSCORE_DELTA
COUNTER_OFF = HIST_OFF + 8192
KILLER_OFF = COUNTER_OFF + 8192
EVSTACK_OFF = KILLER_OFF + 2 * MAX_PLY
# v10: constants of the third network (weights: NN2_W1G etc., loaded at import).
# The NN2_ prefix is the network form, shared with v9's second network.
NN2_H = 384
NN2_NB = 8
NN2_NH = 8
NN2_IN = NN2_NB * 768
NN2_QA = 510
NN2_QB = 4096
NN2_SCALE_NUM = 53
NN2_SCALE_DEN = 100
NN2_CLAMP = 20000           # far from NO_EVAL and from mate scores
I32_LEN = EVSTACK_OFF + MAX_PLY
NO_EVAL = -30000

# i8 buffer: TT depths | TT flags | make/unmake capture record
TT_FLAG_OFF = TT_SIZE
UNDO_OFF = 2 * TT_SIZE
I8_LEN = UNDO_OFF + MAX_PLY

# i64 buffer: search control | SEE swap list
C_NODES, C_ABORT, C_DEADLINE, C_ROOT, C_SELDEPTH, C_REPN = 0, 1, 2, 3, 4, 5
SEE_OFF = 8
I64_LEN = SEE_OFF + 40


def new_state():
    u64 = np.zeros(U64_LEN, dtype=np.uint64)
    i32 = np.zeros(I32_LEN, dtype=np.int32)
    i8 = np.zeros(I8_LEN, dtype=np.int8)
    i8[:TT_SIZE] = -1          # TT depth sentinel: -1 means empty
    i64 = np.zeros(I64_LEN, dtype=np.int64)
    return u64, i32, i8, i64
