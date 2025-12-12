"""Entry point for Battle Gomoku AI matches. Load config, wire players, start Omokgame."""

import yaml
from pathlib import Path

try:
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
except ImportError:
    from Battle_Omok_AI.utils.cli import parse_args
    from Battle_Omok_AI.utils.logger import log_event
    from Battle_Omok_AI.Omokgame import Omokgame
    from Battle_Omok_AI.Iot_20203078_KKR import Iot_20203078_KKR
    from Battle_Omok_AI.Iot_20203078_GIR import Iot_20203078_GIR
    from Battle_Omok_AI.Player import HumanPlayer, GuiHumanPlayer
    from Battle_Omok_AI.ai import heuristic
    from Battle_Omok_AI.gui.pygame_view import PygameView
    from Battle_Omok_AI.ai import search_mcts, search_minimax
    from Battle_Omok_AI.ai import policy_value


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repo-relative path when invoked from outside `Battle_Omok_AI/`."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = PROJECT_DIR / p
    return candidate if candidate.exists() else p


def load_settings(path):
    path = resolve_project_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    settings = load_settings(args.settings)

    board_size = args.board_size or settings.get("board_size", 15)
    move_timeout = args.timeout or settings.get("move_timeout_seconds", 5)
    depth = args.depth or settings.get("search_depth", 3)
    candidate_limit = args.candidate_limit or settings.get("candidate_limit", 15)
    search_backend = args.search_backend or settings.get("search_backend", "minimax")
    rollout_limit = args.rollout_limit or settings.get("rollout_limit", 512)
    explore = args.explore or settings.get("explore", 1.4)
    dirichlet_alpha = args.dirichlet_alpha if args.dirichlet_alpha is not None else settings.get("dirichlet_alpha", 0.0)
    dirichlet_frac = args.dirichlet_frac if args.dirichlet_frac is not None else settings.get("dirichlet_frac", 0.0)
    default_temp = 1e-6 if search_backend == "mcts" else 1.0
    temperature = args.temperature if args.temperature is not None else settings.get("temperature", default_temp)

    patterns = heuristic.load_patterns()

    # Optional PV checkpoint
    pv_path = resolve_project_path(args.pv_checkpoint) if args.pv_checkpoint else resolve_project_path("checkpoints/pv_latest.pt")
    pv_device = args.pv_device or "cpu"
    pv_helper = None
    if pv_path.exists():
        try:
            pv_helper = policy_value.PolicyValueInfer(str(pv_path), device=pv_device)
            if getattr(pv_helper, "board_size", None) not in (None, board_size) and pv_helper.board_size != board_size:
                print(
                    f"Warning: PV checkpoint board_size={pv_helper.board_size} does not match "
                    f"current board_size={board_size}; proceeding without PV model."
                )
                pv_helper = None
            else:
                search_mcts.PV_HELPER = pv_helper
                print(f"Loaded PV checkpoint: {pv_path} (device={pv_device})")
        except Exception as exc:
            print(f"Warning: Failed to load PV checkpoint at {pv_path}: {exc}; proceeding without PV model.")
            pv_helper = None
    else:
        if args.pv_checkpoint:
            print(f"Warning: PV checkpoint not found at {pv_path}, proceeding without PV model.")

    # VCF option
    enable_vcf = bool(getattr(args, "enable_vcf", False))
    search_args = {
        "rollout_limit": rollout_limit,
        "candidate_limit": candidate_limit,
        "explore": explore,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_frac": dirichlet_frac,
        "temperature": temperature,
    }

    black: object
    white: object
    view = None
    if args.gui:
        view = PygameView(board_size=board_size, asset_dir=str(resolve_project_path("assets")))

    if args.mode == "ai-vs-ai":
        black = Iot_20203078_KKR(
            color=-1,
            depth=depth,
            candidate_limit=candidate_limit,
            patterns=patterns,
            pv_helper=pv_helper,
            enable_vcf=enable_vcf,
            search_backend=search_backend,
            search_args=search_args.copy(),
        )
        white = Iot_20203078_GIR(
            color=1,
            depth=depth,
            candidate_limit=candidate_limit,
            patterns=patterns,
            pv_helper=pv_helper,
            enable_vcf=enable_vcf,
            search_backend=search_backend,
            search_args=search_args.copy(),
        )
    elif args.mode == "human-vs-ai":
        if view:
            black = GuiHumanPlayer(color=-1, view=view)
        else:
            black = HumanPlayer(color=-1)
        white = Iot_20203078_GIR(
            color=1,
            depth=depth,
            candidate_limit=candidate_limit,
            patterns=patterns,
            pv_helper=pv_helper,
            enable_vcf=enable_vcf,
            search_backend=search_backend,
            search_args=search_args.copy(),
        )
    elif args.mode == "ai-vs-human":
        black = Iot_20203078_KKR(
            color=-1,
            depth=depth,
            candidate_limit=candidate_limit,
            patterns=patterns,
            pv_helper=pv_helper,
            enable_vcf=enable_vcf,
            search_backend=search_backend,
            search_args=search_args.copy(),
        )
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
