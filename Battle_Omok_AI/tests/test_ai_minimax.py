"""Tests for minimax move selection."""

import time

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.ai import heuristic, search_minimax


def test_minimax_prefers_immediate_win_and_does_not_mutate_board():
    b = Board(size=5)
    for x in range(4):
        b.place(x, 0, -1)

    before_cells = [row[:] for row in b.cells]
    before_count = b.move_count

    mv = search_minimax.choose_move(
        b,
        color=-1,
        depth=1,
        deadline=time.time() + 0.5,
        candidate_limit=25,
        patterns=heuristic.DEFAULT_PATTERNS,
    )

    assert mv == (4, 0)
    assert b.cells == before_cells
    assert b.move_count == before_count
    assert b.is_empty(*mv)
