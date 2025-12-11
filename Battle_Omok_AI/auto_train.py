"""Iterative self-play -> PV train loop for reinforcement learning style updates."""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(f"--> {cmd}")
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Iterative self-play + PV train loop")
    parser.add_argument("--iterations", type=int, default=10, help="Number of self-play/train cycles")
    parser.add_argument("--games", type=int, default=200, help="Number of self-play games per iteration")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data", default="selfplay_renju.jsonl", help="Base path for self-play data (will be suffixed per iteration)")
    parser.add_argument("--checkpoint", default="checkpoints/pv_latest.pt", help="Path to save/load PV checkpoint")
    parser.add_argument("--device", default="cpu", help="Device for PV inference during self-play (cpu or cuda)")
    parser.add_argument("--swap-colors", action="store_true", help="Swap black/white each game for diversity")
    parser.add_argument("--random-open", type=int, default=0, help="Random legal moves for first N plies")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon-greedy random move probability")
    parser.add_argument("--black-baseline", choices=["none", "random", "greedy"], default="none", help="Use baseline for black (evaluation)")
    parser.add_argument("--white-baseline", choices=["none", "random", "greedy"], default="none", help="Use baseline for white (evaluation)")
    parser.add_argument("--collect-stats", action="store_true", help="Collect per-move search stats during self-play")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root exploration noise")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="Dirichlet mix fraction at root")
    parser.add_argument("--temperature", type=float, default=1.0, help="Root visit temperature for move sampling")
    args = parser.parse_args()

    cwd = Path(__file__).parent
    base = Path(args.data)
    stem = base.stem
    suffix = base.suffix or ".jsonl"
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    base.parent.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.iterations + 1):
        print(f"\n=== Iteration {i}/{args.iterations} ===")
        data_file = base.with_name(f"{stem}_iter_{i}{suffix}")

        # 1) self-play using latest checkpoint (if exists)
        ckpt_exists = Path(args.checkpoint).exists()
        selfplay_cmd = (
            f"{sys.executable} selfplay.py --games {args.games} --board-size {args.board_size} "
            f"--depth {args.depth} --candidate-limit {args.candidate_limit} --timeout {args.timeout} "
            f"--output {data_file}"
            f"{' --swap-colors' if args.swap_colors else ''}"
            f"{' --random-open ' + str(args.random_open) if args.random_open else ''}"
            f"{' --epsilon ' + str(args.epsilon) if args.epsilon > 0 else ''}"
            f"{' --black-baseline ' + args.black_baseline if args.black_baseline != 'none' else ''}"
            f"{' --white-baseline ' + args.white_baseline if args.white_baseline != 'none' else ''}"
            f"{' --collect-stats' if args.collect_stats else ''}"
            f"{' --dirichlet-alpha ' + str(args.dirichlet_alpha) if args.dirichlet_alpha is not None else ''}"
            f"{' --dirichlet-frac ' + str(args.dirichlet_frac) if args.dirichlet_frac is not None else ''}"
            f"{' --temperature ' + str(args.temperature) if args.temperature is not None else ''}"
            f"{' --pv-checkpoint ' + str(args.checkpoint) if ckpt_exists else ''}"
            f"{' --pv-device ' + args.device if ckpt_exists else ''}"
        )
        run(selfplay_cmd, cwd=cwd)

        # 2) train PV on newly generated data
        train_cmd = (
            f"{sys.executable} train_pv.py --data {data_file} --board-size {args.board_size} "
            f"--epochs {args.epochs} --batch-size {args.batch_size} --output {args.checkpoint}"
        )
        run(train_cmd, cwd=cwd)

        print(f"=== Completed iteration {i}/{args.iterations} ===")


if __name__ == "__main__":
    main()
