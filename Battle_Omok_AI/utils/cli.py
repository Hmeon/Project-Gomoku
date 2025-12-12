"""CLI options for selecting players, board size, and config paths."""


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Battle Gomoku AI (Renju Rules)")
    parser.add_argument("--board-size", type=int, help="Board size (15 or 19)")
    parser.add_argument("--timeout", type=float, help="Seconds per move (default from settings)")
    parser.add_argument("--depth", type=int, help="Search depth for AI")
    parser.add_argument("--candidate-limit", type=int, help="Number of candidate moves to expand")
    parser.add_argument(
        "--mode",
        choices=["ai-vs-ai", "human-vs-ai", "ai-vs-human", "human-vs-human"],
        default="ai-vs-ai",
        help="Play mode (who plays black/white)",
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--gui", action="store_true", help="Enable pygame GUI (mouse input for human)")
    parser.add_argument("--pv-checkpoint", help="Path to policy/value checkpoint to load (optional)")
    parser.add_argument("--pv-device", default=None, help="Device for PV model (cpu or cuda, optional)")
    parser.add_argument("--enable-vcf", action="store_true", help="Enable VCF search (slower but finds forced wins)")
    parser.add_argument(
        "--search-backend",
        choices=["minimax", "mcts"],
        default=None,
        help="Search backend for AI players (default from settings or minimax)",
    )
    parser.add_argument("--rollout-limit", type=int, default=None, help="MCTS rollouts per move")
    parser.add_argument("--explore", type=float, default=None, help="PUCT exploration constant c_puct for MCTS")
    parser.add_argument("--dirichlet-alpha", type=float, default=None, help="Dirichlet alpha for MCTS root noise")
    parser.add_argument("--dirichlet-frac", type=float, default=None, help="Dirichlet noise mix fraction at root")
    parser.add_argument("--temperature", type=float, default=None, help="Root visit temperature for MCTS sampling")
    return parser.parse_args()
