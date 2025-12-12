"""Game loop and turn management for Gomoku-Pro rule set."""

import time
try:
    from Board import Board
    from engine import referee, renju_rules
    from utils import timer
except ImportError:
    from Battle_Omok_AI.Board import Board
    from Battle_Omok_AI.engine import referee, renju_rules
    from Battle_Omok_AI.utils import timer


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
        game_result = None
        last_move = None
        try:
            while game_result is None:
                if self.renderer:
                    self.renderer(self.board, last_move, color, game_result)

                player = self.players[color]
                deadline = timer.deadline_after(self.move_timeout)

                try:
                    move = player.next_move(self.board, deadline=deadline)
                    referee.check_move(move, self.board, color, deadline, move_index=self.move_index)
                    self.board.place(*move, color)
                    last_move = move
                except (TimeoutError, ValueError) as exc:
                    self.logger(f"Disqualification: {'Black' if color == -1 else 'White'} - {exc}")
                    game_result = -color  # opponent wins
                    break

                self.logger(f"Move {self.move_index + 1}: {'B' if color == -1 else 'W'} {move}")

                if renju_rules.is_win_after_move(self.board, *move, color):
                    self.logger(f"Winner: {'Black' if color == -1 else 'White'}")
                    game_result = color
                elif self.board.move_count >= self.board.size * self.board.size:
                    self.logger("Result: Draw (board full)")
                    game_result = 0
                
                color = -color  # swap turns
                self.move_index += 1

            if self.renderer:
                self.renderer(self.board, last_move, color, game_result)
                # Pause to show the result
                time.sleep(3)
            
            return game_result
        finally:
            if self.closer:
                self.closer()
