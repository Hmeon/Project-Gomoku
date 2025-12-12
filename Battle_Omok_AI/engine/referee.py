"""Move validation, time control, and disqualification handling."""

import time

try:
    from . import renju_rules
except ImportError:
    from engine import renju_rules


def check_move(move, board, color, deadline, move_index=None):
    """
    Validate a move against time, bounds, occupancy, and Renju forbidden-move rules.
    Raises ValueError/TimeoutError on invalid moves.
    """
    if time.time() > deadline:
        raise TimeoutError("Move exceeded allotted time")

    x, y = move
    if not board.in_bounds(x, y):
        raise ValueError("Move out of bounds")
    if not board.is_empty(x, y):
        raise ValueError("Cell already occupied")

    if renju_rules.is_forbidden(board, x, y, color):
        raise ValueError("Forbidden move (Renju rule)")

    return True
