"""Search-level tests for MCTS fallback and TT key shape."""

import time

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.ai import heuristic, search_mcts, search_minimax, transposition


def test_mcts_returns_legal_move():
    b = Board(size=9)
    b.place(4, 4, -1)
    b.place(4, 5, 1)
    deadline = time.time() + 0.5
    mv = search_mcts.choose_move(b, color=-1, deadline=deadline, rollout_limit=64, candidate_limit=20)
    assert b.in_bounds(*mv)
    assert b.is_empty(*mv)


def test_transposition_key_includes_color():
    b = Board(size=5)
    b.place(2, 2, -1)
    ztable = transposition.zobrist_init(b.size)
    cache = {}
    search_minimax.choose_move(
        b,
        color=-1,
        depth=1,
        deadline=time.time() + 1,
        cache=cache,
        zobrist_table=ztable,
        candidate_limit=10,
        patterns=heuristic.DEFAULT_PATTERNS,
    )
    assert cache, "Expected TT cache to be populated"
    for key in cache.keys():
        assert isinstance(key, tuple) and len(key) == 2
        assert key[1] in (-1, 1)
