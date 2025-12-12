"""Tests for Omokgame turn handling and end-of-game state."""

import pytest

from Battle_Omok_AI.Omokgame import Omokgame
from Battle_Omok_AI.Player import Player


class SeqPlayer(Player):
    """Deterministic player that plays a fixed move sequence."""

    def __init__(self, color, moves):
        super().__init__(color)
        self._moves = list(moves)
        self._idx = 0

    def next_move(self, board, deadline=None):
        if self._idx >= len(self._moves):
            raise ValueError("No more scripted moves")
        mv = self._moves[self._idx]
        self._idx += 1
        return mv


def test_final_render_color_is_winner(monkeypatch):
    # Avoid 3s GUI pause in Omokgame when renderer is set.
    import importlib

    omok_mod = importlib.import_module("Battle_Omok_AI.Omokgame")

    monkeypatch.setattr(omok_mod.time, "sleep", lambda *_: None)

    black_moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    white_moves = [(0, 1), (1, 1), (2, 1), (3, 1)]

    black = SeqPlayer(-1, black_moves)
    white = SeqPlayer(1, white_moves)

    final_colors = []

    def renderer(board, last_move, current_color, game_result):
        if game_result is not None:
            final_colors.append(current_color)

    game = Omokgame(
        board_size=5,
        move_timeout=5.0,
        black_player=black,
        white_player=white,
        renderer=renderer,
    )
    result = game.play()

    assert result == -1
    assert final_colors, "Expected renderer to be called with final state"
    assert final_colors[-1] == -1
