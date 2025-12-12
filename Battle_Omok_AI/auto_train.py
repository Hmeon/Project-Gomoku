"""Iterative self-play -> PV train loop for reinforcement learning style updates."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def run(cmd, cwd=None):
    printable = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    print(f"--> {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def log_selfplay_stats(iteration, stats_file):
    """Read stats JSON and append to CSV log."""
    stats_path = Path(stats_file)
    if not stats_path.exists():
        print(f"Warning: Stats file {stats_file} not found.")
        return

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    log_dir = ROOT / "logs"
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


def log_eval_metrics(
    iteration: int,
    *,
    games: int,
    wins: int,
    draws: int,
    score_rate: float | None,
    threshold: float | None,
    decision: str,
    eval_stats_file: Path | None,
    candidate_ckpt: Path | None,
    incumbent_ckpt: Path | None,
) -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_csv = log_dir / "eval_metrics.csv"
    file_exists = log_csv.exists()

    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "iteration",
                    "games",
                    "candidate_wins",
                    "candidate_draws",
                    "score_rate",
                    "threshold",
                    "decision",
                    "eval_stats_file",
                    "candidate_ckpt",
                    "incumbent_ckpt",
                ]
            )

        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                iteration,
                games,
                wins,
                draws,
                f"{score_rate:.4f}" if score_rate is not None else "",
                f"{threshold:.4f}" if threshold is not None else "",
                decision,
                str(eval_stats_file) if eval_stats_file is not None else "",
                str(candidate_ckpt) if candidate_ckpt is not None else "",
                str(incumbent_ckpt) if incumbent_ckpt is not None else "",
            ]
        )


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
    parser.add_argument("--train-workers", type=int, default=0, help="DataLoader workers for training (0 is safest on Windows)")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in training DataLoader (useful for CUDA)")
    parser.add_argument("--data", default="selfplay_renju.jsonl", help="Base path for self-play data (will be suffixed per iteration)")
    parser.add_argument("--checkpoint", default="checkpoints/pv_latest.pt", help="Path to save/load PV checkpoint")
    parser.add_argument("--replay-window", type=int, default=1, help="Train on the most recent N self-play files (default 1)")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply random rotations/flips during training (recommended).",
    )
    parser.add_argument("--augment-prob", type=float, default=1.0, help="Symmetry augmentation probability per sample")
    parser.add_argument("--seed", type=int, default=None, help="Seed for self-play randomness and training augmentation (optional)")
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
    parser.add_argument("--search-backend", choices=["minimax", "mcts"], default="minimax", help="Search algorithm to use during self-play")
    parser.add_argument("--rollout-limit", type=int, default=512, help="MCTS rollouts per move (only for --search-backend mcts)")
    parser.add_argument("--explore", type=float, default=1.4, help="PUCT exploration constant c_puct (only for --search-backend mcts)")
    parser.add_argument("--eval-games", type=int, default=0, help="If >0, evaluate candidate vs current checkpoint before promotion")
    parser.add_argument("--accept-threshold", type=float, default=0.55, help="Candidate score rate needed to replace current checkpoint")
    args = parser.parse_args()

    cwd = ROOT
    base = _resolve_under_root(args.data)
    stem = base.stem
    suffix = base.suffix or ".jsonl"
    ckpt_path = _resolve_under_root(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    base.parent.mkdir(parents=True, exist_ok=True)

    def promote_checkpoint(candidate: Path, target: Path, *, backup_suffix: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_name(target.stem + backup_suffix + target.suffix)
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.replace(backup)
        candidate.replace(target)

    def gather_replay_files(iteration: int) -> list[Path]:
        window = max(int(args.replay_window), 1)
        start = max(1, iteration - window + 1)
        files: list[Path] = []
        for j in range(start, iteration + 1):
            p = base.with_name(f"{stem}_iter_{j}{suffix}")
            if p.exists():
                files.append(p)
        return files or [base.with_name(f"{stem}_iter_{iteration}{suffix}")]

    for i in range(1, args.iterations + 1):
        print(f"\n=== Iteration {i}/{args.iterations} ===")
        data_file = base.with_name(f"{stem}_iter_{i}{suffix}")

        # 1) self-play using latest checkpoint (if exists)
        ckpt_exists = ckpt_path.exists()
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
            "--search-backend",
            args.search_backend,
        ]
        if args.seed is not None:
            selfplay_cmd += ["--seed", str(args.seed + i)]
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
        if args.search_backend == "mcts":
            selfplay_cmd += ["--rollout-limit", str(args.rollout_limit), "--explore", str(args.explore)]
        if ckpt_exists:
            selfplay_cmd += ["--pv-checkpoint", str(args.checkpoint), "--pv-device", args.device]
        if args.enable_vcf:
            selfplay_cmd.append("--enable-vcf")
        run(selfplay_cmd, cwd=cwd)

        # Log self-play stats
        stats_file = data_file.with_name(data_file.stem + "_stats.json")
        log_selfplay_stats(i, stats_file)

        # 2) train PV on newly generated data
        candidate_ckpt = ckpt_path.with_name(ckpt_path.stem + "_candidate" + ckpt_path.suffix)
        replay_files = gather_replay_files(i)
        train_cmd = [
            sys.executable,
            "train_pv.py",
            "--data",
            *[str(p) for p in replay_files],
            "--board-size",
            str(args.board_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.train_workers),
            "--output",
            str(candidate_ckpt),
            "--device",
            args.device,
        ]
        train_cmd += ["--augment" if args.augment else "--no-augment", "--augment-prob", str(args.augment_prob)]
        if args.pin_memory:
            train_cmd.append("--pin-memory")
        if args.seed is not None:
            train_cmd += ["--seed", str(args.seed + i)]
        run(train_cmd, cwd=cwd)

        # 3) optional evaluation gate: only promote if candidate beats current
        if ckpt_exists and args.eval_games > 0:
            eval_out = ROOT / "logs" / f"eval_iter_{i}.jsonl"
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_cmd = [
                sys.executable,
                "selfplay.py",
                "--stats-only",
                "--games",
                str(args.eval_games),
                "--board-size",
                str(args.board_size),
                "--depth",
                str(args.depth),
                "--candidate-limit",
                str(args.candidate_limit),
                "--timeout",
                str(args.timeout),
                "--output",
                str(eval_out),
                "--search-backend",
                args.search_backend,
                "--swap-colors",
                "--random-open",
                "0",
                "--epsilon",
                "0",
                "--dirichlet-alpha",
                "0.0",
                "--dirichlet-frac",
                "0.0",
                "--temperature",
                "1e-6",
                "--black-tag",
                "candidate",
                "--white-tag",
                "incumbent",
                "--black-pv-checkpoint",
                str(candidate_ckpt),
                "--white-pv-checkpoint",
                str(ckpt_path),
                "--pv-device",
                args.device,
            ]
            if args.seed is not None:
                eval_cmd += ["--seed", str(args.seed + 10_000 + i)]
            if args.search_backend == "mcts":
                eval_cmd += ["--rollout-limit", str(args.rollout_limit), "--explore", str(args.explore)]
            if args.enable_vcf:
                eval_cmd.append("--enable-vcf")
            run(eval_cmd, cwd=cwd)

            eval_stats_file = eval_out.with_name(eval_out.stem + "_stats.json")
            with open(eval_stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            games = max(int(stats.get("games", 0)), 1)
            wins = stats.get("agent_wins", {}).get("candidate", 0)
            draws = stats.get("agent_draws", {}).get("candidate", 0)
            score_rate = (wins + 0.5 * draws) / games

            print(
                f"Eval candidate vs incumbent: wins={wins}, draws={draws}, games={games}, "
                f"score_rate={score_rate:.3f} (threshold={args.accept_threshold:.3f})"
            )

            if score_rate >= args.accept_threshold:
                backup = ckpt_path.with_name(ckpt_path.stem + "_prev" + ckpt_path.suffix)
                promote_checkpoint(candidate_ckpt, ckpt_path, backup_suffix="_prev")
                print("Promoted candidate checkpoint.")
                log_eval_metrics(
                    i,
                    games=games,
                    wins=wins,
                    draws=draws,
                    score_rate=score_rate,
                    threshold=args.accept_threshold,
                    decision="promoted",
                    eval_stats_file=eval_stats_file,
                    candidate_ckpt=ckpt_path,
                    incumbent_ckpt=backup,
                )
            else:
                rejected = ckpt_path.with_name(f"{ckpt_path.stem}_rejected_iter_{i}{ckpt_path.suffix}")
                candidate_ckpt.replace(rejected)
                print(f"Rejected candidate checkpoint; saved to {rejected}.")
                log_eval_metrics(
                    i,
                    games=games,
                    wins=wins,
                    draws=draws,
                    score_rate=score_rate,
                    threshold=args.accept_threshold,
                    decision="rejected",
                    eval_stats_file=eval_stats_file,
                    candidate_ckpt=rejected,
                    incumbent_ckpt=ckpt_path,
                )
        else:
            backup = None
            if ckpt_path.exists():
                backup = ckpt_path.with_name(ckpt_path.stem + "_prev" + ckpt_path.suffix)
            promote_checkpoint(candidate_ckpt, ckpt_path, backup_suffix="_prev")
            log_eval_metrics(
                i,
                games=0,
                wins=0,
                draws=0,
                score_rate=None,
                threshold=None,
                decision="bootstrap" if not ckpt_exists else "promoted_no_eval",
                eval_stats_file=None,
                candidate_ckpt=ckpt_path,
                incumbent_ckpt=backup,
            )

        print(f"=== Completed iteration {i}/{args.iterations} ===")


if __name__ == "__main__":
    main()
