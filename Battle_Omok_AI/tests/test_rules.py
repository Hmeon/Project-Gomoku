"""Renju rule checks: forbidden for Black and allowed for White."""

import pytest

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.engine import renju_rules


def test_black_overline_forbidden():
    b = Board(size=15)
    # Create a line of 5; adding one more makes overline with no simultaneous five elsewhere.
    for x in range(5):
        b.place(x, 7, -1)
    assert renju_rules.is_forbidden(b, 5, 7, -1)


def test_black_double_three_forbidden():
    b = Board(size=15)
    # Shape that yields two open threes when placing at center.
    stones = [(7, 6), (7, 8), (6, 7), (8, 7)]
    for x, y in stones:
        b.place(x, y, -1)
    assert renju_rules.is_forbidden(b, 7, 7, -1)


def test_black_double_four_forbidden():
    b = Board(size=15)
    # Two open-fours simultaneously when placing at (7,7)
    coords = [
        # horizontal three with a gap at (7,7)
        (5, 7), (6, 7), (8, 7),
        # vertical three with a gap at (7,7)
        (7, 5), (7, 6), (7, 8),
    ]
    for x, y in coords:
        b.place(x, y, -1)
    assert renju_rules.is_forbidden(b, 7, 7, -1)


def test_black_open_four_allowed():
    b = Board(size=15)
    # Creating a single open four is legal for black (not a 4-4 foul).
    for x in (5, 6, 7):
        b.place(x, 7, -1)
    assert renju_rules.is_forbidden(b, 8, 7, -1) is False


def test_five_trumps_foul():
    b = Board(size=15)
    # Completing vertical five while also forming horizontal fours; should be allowed.
    for y in range(2, 6):
        b.place(7, y, -1)
    b.place(6, 7, -1)
    b.place(8, 7, -1)
    assert renju_rules.is_forbidden(b, 7, 6, -1) is False  # move makes exact five vertically
    with renju_rules._simulate(b, 7, 6, -1):
        assert renju_rules.is_win_after_move(b, 7, 6, -1)


def test_white_wins_with_six():
    b = Board(size=15)
    for x in range(5):
        b.place(x, 3, 1)
    b.place(5, 3, 1)
    assert renju_rules.is_win_after_move(b, 5, 3, 1)
