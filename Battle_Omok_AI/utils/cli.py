"""CLI options for selecting players, board size, and config paths."""


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Battle Gomoku-Pro AI")
    parser.add_argument("--board-size", type=int, help="Board size (15 or 19)")
    parser.add_argument("--timeout", type=float, help="Seconds per move (default from settings)")
    parser.add_argument("--depth", type=int, help="Search depth for AI")
    parser.add_argument("--candidate-limit", type=int, help="Number of candidate moves to expand")
    parser.add_argument(
        "--mode",
        choices=["ai-vs-ai", "human-vs-ai", "ai-vs-human"],
        default="ai-vs-ai",
        help="Play mode (who plays black/white)",
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--gui", action="store_true", help="Enable pygame GUI (mouse input for human)")
    parser.add_argument("--pv-checkpoint", help="Path to policy/value checkpoint to load (optional)")
    parser.add_argument("--pv-device", default=None, help="Device for PV model (cpu or cuda, optional)")
    return parser.parse_args()
