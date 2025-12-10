"""Sanity tests for Board exact-five logic and placement validity."""

from Battle_Omok_AI.Board import Board


def test_exact_five_wins_but_overline_does_not():
    b = Board(size=15)
    # place five in a row horizontally
    moves = [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]
    for x, y in moves:
        b.place(x, y, -1)
    assert b.has_exact_five(5, 5)
    # add one more to create overline
    b.place(8, 5, -1)
    assert not b.has_exact_five(5, 5)
    assert not b.has_exact_five(8, 5)
