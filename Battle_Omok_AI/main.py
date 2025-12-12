"""Entry point for Battle Gomoku AI matches. Load config, wire players, start Omokgame."""

import yaml
from pathlib import Path

from utils.cli import parse_args
from utils.logger import log_event
from Omokgame import Omokgame
from Iot_20203078_KKR import Iot_20203078_KKR
from Iot_20203078_GIR import Iot_20203078_GIR
from Player import HumanPlayer, GuiHumanPlayer
from ai import heuristic
from gui.pygame_view import PygameView
from ai import search_mcts, search_minimax
from ai import policy_value


def load_settings(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    settings = load_settings(args.settings)

    board_size = args.board_size or settings.get("board_size", 15)
    move_timeout = args.timeout or settings.get("move_timeout_seconds", 5)
    depth = args.depth or settings.get("search_depth", 3)
    candidate_limit = args.candidate_limit or settings.get("candidate_limit", 15)

    patterns = heuristic.load_patterns()

    # Optional PV checkpoint
    pv_path = Path(args.pv_checkpoint) if args.pv_checkpoint else Path("checkpoints/pv_latest.pt")
    pv_device = args.pv_device or "cpu"
    pv_helper = None
    if pv_path.exists():
        pv_helper = policy_value.PolicyValueInfer(str(pv_path), device=pv_device)
        search_mcts.PV_HELPER = pv_helper
        print(f"Loaded PV checkpoint: {pv_path} (device={pv_device})")
    else:
        if args.pv_checkpoint:
            print(f"Warning: PV checkpoint not found at {pv_path}, proceeding without PV model.")

    # VCF option
    enable_vcf = args.enable_vcf if hasattr(args, 'enable_vcf') else False

    black: object
    white: object
    view = None
    if args.gui:
        view = PygameView(board_size=board_size, asset_dir="assets")

    if args.mode == "ai-vs-ai":
        black = Iot_20203078_KKR(color=-1, depth=depth, candidate_limit=candidate_limit, patterns=patterns, pv_helper=pv_helper, enable_vcf=enable_vcf)
        white = Iot_20203078_GIR(color=1, depth=depth, candidate_limit=candidate_limit, patterns=patterns, pv_helper=pv_helper, enable_vcf=enable_vcf)
    elif args.mode == "human-vs-ai":
        if view:
            black = GuiHumanPlayer(color=-1, view=view)
        else:
            black = HumanPlayer(color=-1)
        white = Iot_20203078_GIR(color=1, depth=depth, candidate_limit=candidate_limit, patterns=patterns, pv_helper=pv_helper, enable_vcf=enable_vcf)
    elif args.mode == "ai-vs-human":
        black = Iot_20203078_KKR(color=-1, depth=depth, candidate_limit=candidate_limit, patterns=patterns, pv_helper=pv_helper, enable_vcf=enable_vcf)
        if view:
            white = GuiHumanPlayer(color=1, view=view)
        else:
            white = HumanPlayer(color=1)
    elif args.mode == "human-vs-human":
        if view:
            black = GuiHumanPlayer(color=-1, view=view)
            white = GuiHumanPlayer(color=1, view=view)
        else:
            black = HumanPlayer(color=-1)
            white = HumanPlayer(color=1)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    game = Omokgame(
        board_size=board_size,
        move_timeout=move_timeout,
        black_player=black,
        white_player=white,
        logger=log_event,
        patterns=patterns,
        renderer=view.render if view else None,
        closer=view.close if view else None,
    )
    result = game.play()
    outcome = { -1: "Black wins", 1: "White wins", 0: "Draw" }
    print(outcome.get(result, "Unknown result"))


if __name__ == "__main__":
    main()
