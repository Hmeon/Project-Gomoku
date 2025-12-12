"""Incremental heuristic update should match full recomputation."""

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.ai import heuristic


def test_incremental_matches_full():
    b = Board(size=5)
    eval_color = -1
    base = heuristic.score_board(b, eval_color)

    b.place(1, 1, -1)
    full = heuristic.score_board(b, eval_color)
    inc = heuristic.update_score_after_move(
        b, 1, 1, -1, eval_color, base, patterns=heuristic.DEFAULT_PATTERNS
    )
    assert full == inc

    b.place(2, 2, 1)
    full2 = heuristic.score_board(b, eval_color)
    inc2 = heuristic.update_score_after_move(
        b, 2, 2, 1, eval_color, inc, patterns=heuristic.DEFAULT_PATTERNS
    )
    assert full2 == inc2
