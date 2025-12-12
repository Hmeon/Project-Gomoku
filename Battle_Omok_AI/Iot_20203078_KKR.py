"""Black-player AI implementation."""

try:
    from Player import Player
    from ai import search_minimax, transposition, search_mcts
except ImportError:
    from Battle_Omok_AI.Player import Player
    from Battle_Omok_AI.ai import search_minimax, transposition, search_mcts


class Iot_20203078_KKR(Player):
    def __init__(self, color=-1, depth=3, candidate_limit=15, patterns=None, pv_helper=None, enable_vcf=False, search_backend="minimax", search_args=None):
        super().__init__(color)
        self.depth = depth
        self.candidate_limit = candidate_limit
        self.cache = {}
        self.zobrist = None
        self.patterns = patterns
        self.pv_helper = pv_helper
        self.enable_vcf = enable_vcf
        self.search_args = search_args or {}
        self.search_backend = search_backend

    def next_move(self, board, deadline=None):
        if self.zobrist is None:
            self.zobrist = transposition.zobrist_init(board.size)

        if self.search_backend == "mcts":
            args = self.search_args
            return search_mcts.choose_move(
                board,
                color=self.color,
                deadline=deadline,
                rollout_limit=args.get("rollout_limit", 512),
                candidate_limit=args.get("candidate_limit", self.candidate_limit),
                explore=args.get("explore", 1.4),
                dirichlet_alpha=args.get("dirichlet_alpha", 0.3),
                dirichlet_frac=args.get("dirichlet_frac", 0.25),
                temperature=args.get("temperature", 1.0),
                pv_helper=self.pv_helper,
            )

        return search_minimax.choose_move(
            board,
            self.color,
            depth=self.depth,
            deadline=deadline,
            cache=self.cache,
            zobrist_table=self.zobrist,
            candidate_limit=self.candidate_limit,
            patterns=self.patterns,
            pv_helper=self.pv_helper,
            enable_vcf=self.enable_vcf,
            stats=getattr(self, "stats", None),
        )
