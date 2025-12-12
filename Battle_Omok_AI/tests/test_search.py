"""Search-level tests for MCTS fallback and TT key shape."""

import time
import pytest

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


def test_mcts_fallback_when_candidates_empty(monkeypatch):
    # Force candidate generator to return empty so MCTS must use full-scan fallback.
    monkeypatch.setattr(search_mcts.move_selector, "generate_candidates", lambda *args, **kwargs: [])
    b = Board(size=5)
    b.place(2, 2, -1)
    deadline = time.time() + 0.5
    mv = search_mcts.choose_move(b, color=-1, deadline=deadline, rollout_limit=8, candidate_limit=5)
    assert b.in_bounds(*mv)
    assert b.is_empty(*mv)


def test_minimax_fallback_when_candidates_empty(monkeypatch):
    # Force candidate generator to return empty so minimax must use full-scan fallback.
    monkeypatch.setattr(search_minimax.move_selector, "generate_candidates", lambda *args, **kwargs: [])
    b = Board(size=5)
    deadline = time.time() + 1.0
    mv = search_minimax.choose_move(
        b,
        color=-1,
        depth=1,
        deadline=deadline,
        candidate_limit=5,
        patterns=heuristic.DEFAULT_PATTERNS,
    )
    assert b.in_bounds(*mv)
    assert b.is_empty(*mv)


def test_mcts_prefers_immediate_win():
    import random

    random.seed(0)
    b = Board(size=5)
    # Black has four in a row; (4,0) is an exact-five winning move.
    for x in range(4):
        b.place(x, 0, -1)

    deadline = time.time() + 0.5
    mv = search_mcts.choose_move(
        b,
        color=-1,
        deadline=deadline,
        rollout_limit=64,
        candidate_limit=10,
        dirichlet_alpha=0.0,
        dirichlet_frac=0.0,
        temperature=1e-6,
    )
    assert mv == (4, 0)


def test_mcts_does_not_mutate_board():
    b = Board(size=5)
    b.place(2, 2, -1)
    b.place(1, 1, 1)
    before_cells = [row[:] for row in b.cells]
    before_count = b.move_count

    mv = search_mcts.choose_move(
        b,
        color=-1,
        deadline=time.time() + 0.5,
        rollout_limit=32,
        candidate_limit=10,
        dirichlet_alpha=0.0,
        dirichlet_frac=0.0,
        temperature=1e-6,
    )

    assert b.cells == before_cells
    assert b.move_count == before_count
    assert b.is_empty(*mv)


def test_mcts_raises_when_no_legal_moves(monkeypatch):
    # Simulate a position where every empty move is forbidden for black.
    monkeypatch.setattr(search_mcts.renju_rules, "is_forbidden", lambda *args, **kwargs: True)
    b = Board(size=5)
    with pytest.raises(ValueError):
        search_mcts.choose_move(b, color=-1, deadline=time.time() + 0.1, rollout_limit=8, candidate_limit=5)


def test_minimax_raises_when_no_legal_moves(monkeypatch):
    monkeypatch.setattr(search_minimax.renju_rules, "is_forbidden", lambda *args, **kwargs: True)
    b = Board(size=5)
    with pytest.raises(ValueError):
        search_minimax.choose_move(
            b,
            color=-1,
            depth=1,
            deadline=time.time() + 1.0,
            candidate_limit=5,
            patterns=heuristic.DEFAULT_PATTERNS,
        )
