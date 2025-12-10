"""Tests for per-move timeouts and referee enforcement."""

import time
import pytest

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.engine import referee


def test_timeout_rejected():
    b = Board(size=15)
    deadline = time.time() - 0.1
    with pytest.raises(TimeoutError):
        referee.check_move((7, 7), b, -1, deadline, move_index=0)


def test_valid_move_passes():
    b = Board(size=15)
    deadline = time.time() + 1
    assert referee.check_move((7, 7), b, -1, deadline, move_index=0) is True
