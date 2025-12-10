"""Black-player AI implementation."""

from Player import Player
from ai import search_minimax, transposition


class Iot_20203078_KKR(Player):
    def __init__(self, color=-1, depth=3, candidate_limit=15, patterns=None):
        super().__init__(color)
        self.depth = depth
        self.candidate_limit = candidate_limit
        self.cache = {}
        self.zobrist = None
        self.patterns = patterns

    def next_move(self, board, deadline=None):
        if self.zobrist is None:
            self.zobrist = transposition.zobrist_init(board.size)
        return search_minimax.choose_move(
            board,
            self.color,
            depth=self.depth,
            deadline=deadline,
            cache=self.cache,
            zobrist_table=self.zobrist,
            candidate_limit=self.candidate_limit,
            patterns=self.patterns,
        )
