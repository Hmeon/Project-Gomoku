"""Game loop and turn management for Gomoku-Pro rule set."""

from Board import Board
from engine import referee, renju_rules
from utils import timer


class Omokgame:
    def __init__(self, board_size, move_timeout, black_player, white_player, logger=print, patterns=None, renderer=None, closer=None):
        self.board = Board(size=board_size)
        self.move_timeout = move_timeout
        self.players = {-1: black_player, 1: white_player}
        self.logger = logger
        self.move_index = 0
        self.patterns = patterns
        self.renderer = renderer
        self.closer = closer

    def play(self):
        """Run a single game. Returns -1 (black win), 1 (white win), or 0 (draw)."""
        color = -1  # black starts
        try:
            if self.renderer:
                self.renderer(self.board, last_move=None)

            while True:
                player = self.players[color]
                deadline = timer.deadline_after(self.move_timeout)

                try:
                    move = player.next_move(self.board, deadline=deadline)
                    referee.check_move(move, self.board, color, deadline, move_index=self.move_index)
                    self.board.place(*move, color)
                except (TimeoutError, ValueError) as exc:
                    self.logger(f"Disqualification: {'Black' if color == -1 else 'White'} - {exc}")
                    return -color  # opponent wins

                self.logger(f"Move {self.move_index + 1}: {'B' if color == -1 else 'W'} {move}")
                if self.renderer:
                    self.renderer(self.board, last_move=move)

                if renju_rules.is_win_after_move(self.board, *move, color):
                    self.logger(f"Winner: {'Black' if color == -1 else 'White'}")
                    return color

                self.move_index += 1
                if self.board.move_count >= self.board.size * self.board.size:
                    self.logger("Result: Draw (board full)")
                    return 0

                color = -color  # swap turns
        finally:
            if self.closer:
                self.closer()
