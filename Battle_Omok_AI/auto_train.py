"""Iterative self-play -> PV train loop for reinforcement learning style updates."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, cwd=None):
    printable = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    print(f"--> {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def log_selfplay_stats(iteration, stats_file):
    """Read stats JSON and append to CSV log."""
    if not Path(stats_file).exists():
        print(f"Warning: Stats file {stats_file} not found.")
        return

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_csv = log_dir / "selfplay_metrics.csv"
    file_exists = log_csv.exists()

    games = stats.get("games", 0)
    if games == 0:
        return

    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "iteration", "games", 
                "black_win_rate", "white_win_rate", "draw_rate", 
                "avg_steps", "avg_black_fouls"
            ])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            iteration,
            games,
            f"{stats['black_wins'] / games:.4f}",
            f"{stats['white_wins'] / games:.4f}",
            f"{stats['draws'] / games:.4f}",
            f"{stats['avg_steps']:.2f}",
            f"{stats['black_fouls'] / games:.4f}"
        ])


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
    parser.add_argument("--no-swap", action="store_true", help="Disable swapping black/white each game (default: swap enabled)")
    parser.add_argument("--random-open", type=int, default=0, help="Random legal moves for first N plies")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon-greedy random move probability (default 0.1)")
    parser.add_argument("--black-baseline", choices=["none", "random", "greedy"], default="none", help="Use baseline for black (evaluation)")
    parser.add_argument("--white-baseline", choices=["none", "random", "greedy"], default="none", help="Use baseline for white (evaluation)")
    parser.add_argument("--collect-stats", action="store_true", help="Collect per-move search stats during self-play")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root exploration noise")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="Dirichlet mix fraction at root")
    parser.add_argument("--temperature", type=float, default=1.0, help="Root visit temperature for move sampling")
    parser.add_argument("--enable-vcf", action="store_true", help="Enable VCF search during self-play")
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
        selfplay_cmd = [
            sys.executable,
            "selfplay.py",
            "--games",
            str(args.games),
            "--board-size",
            str(args.board_size),
            "--depth",
            str(args.depth),
            "--candidate-limit",
            str(args.candidate_limit),
            "--timeout",
            str(args.timeout),
            "--output",
            str(data_file),
        ]
        if not args.no_swap:
            selfplay_cmd.append("--swap-colors")
        if args.random_open:
            selfplay_cmd += ["--random-open", str(args.random_open)]
        if args.epsilon > 0:
            selfplay_cmd += ["--epsilon", str(args.epsilon)]
        if args.black_baseline != "none":
            selfplay_cmd += ["--black-baseline", args.black_baseline]
        if args.white_baseline != "none":
            selfplay_cmd += ["--white-baseline", args.white_baseline]
        if args.collect_stats:
            selfplay_cmd.append("--collect-stats")
        if args.dirichlet_alpha is not None:
            selfplay_cmd += ["--dirichlet-alpha", str(args.dirichlet_alpha)]
        if args.dirichlet_frac is not None:
            selfplay_cmd += ["--dirichlet-frac", str(args.dirichlet_frac)]
        if args.temperature is not None:
            selfplay_cmd += ["--temperature", str(args.temperature)]
        if ckpt_exists:
            selfplay_cmd += ["--pv-checkpoint", str(args.checkpoint), "--pv-device", args.device]
        if args.enable_vcf:
            selfplay_cmd.append("--enable-vcf")
        run(selfplay_cmd, cwd=cwd)

        # Log self-play stats
        stats_file = data_file.with_name(data_file.stem + "_stats.json")
        log_selfplay_stats(i, stats_file)

        # 2) train PV on newly generated data
        train_cmd = [
            sys.executable,
            "train_pv.py",
            "--data",
            str(data_file),
            "--board-size",
            str(args.board_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(args.checkpoint),
            "--device",
            args.device,
        ]
        run(train_cmd, cwd=cwd)

        print(f"=== Completed iteration {i}/{args.iterations} ===")


if __name__ == "__main__":
    main()
