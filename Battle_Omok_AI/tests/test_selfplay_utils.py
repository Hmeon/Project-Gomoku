"""Tests for self-play utilities and board snapshot/restore."""

from Battle_Omok_AI.Board import Board
from Battle_Omok_AI.Player import Player
from Battle_Omok_AI import selfplay


def test_snapshot_restore_preserves_state():
    b = Board(size=5)
    b.place(1, 1, -1)
    b.place(2, 2, 1)
    snap = selfplay.snapshot_board_state(b)

    # Mutate board
    b.cells[1][1] = 0
    b.move_count = 0
    b.history = []

    selfplay.restore_board_state(b, snap)

    assert b.cells[1][1] == -1
    assert b.cells[2][2] == 1
    assert b.move_count == 2
    assert b.history == [(1, 1), (2, 2)]


def test_play_game_fallback_when_candidates_empty(monkeypatch):
    # Force candidate generator to return nothing so selfplay must fall back.
    monkeypatch.setattr(selfplay.move_selector, "generate_candidates", lambda *args, **kwargs: [])

    import random
    random.seed(0)

    black = selfplay.RandomBaseline(-1)
    white = selfplay.RandomBaseline(1)

    winner, traj, info = selfplay.play_game(
        board_size=5,
        black=black,
        white=white,
        timeout=0.01,
        random_open=1,
        epsilon=0.0,
    )

    assert len(traj) > 0
    assert winner in (-1, 0, 1)


class SeqPiPlayer(Player):
    """Scripted player that also exposes a soft policy distribution."""

    def __init__(self, color, moves, pi):
        super().__init__(color)
        self._moves = list(moves)
        self._pi = list(pi)
        self._idx = 0

    def next_move(self, board, deadline=None):
        mv = self._moves[self._idx]
        self._idx += 1
        self.last_pi = self._pi
        return mv


def test_play_game_records_player_pi_when_available():
    board_size = 5
    uniform_pi = [1.0 / (board_size * board_size)] * (board_size * board_size)

    black_moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    white_moves = [(0, 1), (1, 1), (2, 1), (3, 1)]

    black = SeqPiPlayer(-1, black_moves, uniform_pi)
    white = SeqPiPlayer(1, white_moves, uniform_pi)

    winner, traj, _ = selfplay.play_game(
        board_size=board_size,
        black=black,
        white=white,
        timeout=1.0,
        random_open=0,
        epsilon=0.0,
    )

    assert winner == -1
    assert traj[0]["pi"] == uniform_pi


def test_play_game_compact_board_by_default():
    board_size = 5
    uniform_pi = [1.0 / (board_size * board_size)] * (board_size * board_size)

    black_moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    white_moves = [(0, 1), (1, 1), (2, 1), (3, 1)]

    black = SeqPiPlayer(-1, black_moves, uniform_pi)
    white = SeqPiPlayer(1, white_moves, uniform_pi)

    winner, traj, _ = selfplay.play_game(
        board_size=board_size,
        black=black,
        white=white,
        timeout=1.0,
        random_open=0,
        epsilon=0.0,
    )

    assert winner == -1
    assert isinstance(traj[0]["board"], dict)
    assert set(traj[0]["board"].keys()) == {"cells"}


def test_play_game_full_board_when_include_history_enabled():
    board_size = 5
    uniform_pi = [1.0 / (board_size * board_size)] * (board_size * board_size)

    black_moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    white_moves = [(0, 1), (1, 1), (2, 1), (3, 1)]

    black = SeqPiPlayer(-1, black_moves, uniform_pi)
    white = SeqPiPlayer(1, white_moves, uniform_pi)

    winner, traj, _ = selfplay.play_game(
        board_size=board_size,
        black=black,
        white=white,
        timeout=1.0,
        random_open=0,
        epsilon=0.0,
        include_history=True,
    )

    assert winner == -1
    assert isinstance(traj[0]["board"], dict)
    assert "cells" in traj[0]["board"]
    assert "history" in traj[0]["board"]


def test_play_game_ends_when_no_legal_moves(monkeypatch):
    # Force both candidate generation and full-scan legal move finder to return nothing.
    # This simulates "no legal moves" and should end the game cleanly.
    monkeypatch.setattr(selfplay.move_selector, "generate_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(selfplay, "all_legal_moves", lambda *args, **kwargs: [])

    black = selfplay.RandomBaseline(-1)
    white = selfplay.RandomBaseline(1)

    winner, traj, info = selfplay.play_game(
        board_size=5,
        black=black,
        white=white,
        timeout=0.01,
        random_open=0,
        epsilon=0.0,
    )

    assert winner == 1
    assert traj == []


class BadPlayer(Player):
    def next_move(self, board, deadline=None):
        raise ValueError("bad move generator")


def test_play_game_fallback_on_invalid_player(monkeypatch):
    # Ensure invalid move exceptions from a player do not crash self-play.
    monkeypatch.setattr(selfplay.renju_rules, "is_forbidden", lambda *args, **kwargs: False)

    import random
    random.seed(0)

    black = BadPlayer(-1)
    white = selfplay.RandomBaseline(1)

    winner, traj, info = selfplay.play_game(
        board_size=5,
        black=black,
        white=white,
        timeout=0.01,
        random_open=0,
        epsilon=0.0,
    )

    assert winner in (-1, 0, 1)
    assert len(traj) > 0
    assert info["invalid_moves"][-1] > 0
