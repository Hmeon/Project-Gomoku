"""Self-play data generator for Renju rules (policy/value targets)."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import List, Tuple
import random
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Board import Board
from engine import referee, renju_rules
from Iot_20203078_KKR import Iot_20203078_KKR
from Iot_20203078_GIR import Iot_20203078_GIR
from Player import Player
from ai import move_selector, heuristic, policy_value, search_minimax, search_mcts


def snapshot_board_state(board: Board):
    """Return a full snapshot of the board (cells, move_count, history)."""
    return {
        "cells": [row[:] for row in board.cells],
        "move_count": board.move_count,
        "history": board.history[:],
    }


def restore_board_state(board: Board, state):
    """Restore board from snapshot produced by snapshot_board_state."""
    board.cells = [row[:] for row in state["cells"]]
    board.move_count = state["move_count"]
    board.history = state["history"][:]


class RandomBaseline(Player):
    """Random legal move baseline (filters black fouls)."""

    def __init__(self, color):
        super().__init__(color)

    def next_move(self, board, deadline=None):
        candidates = move_selector.generate_candidates(board, limit=board.size * board.size)
        legal = candidates
        if self.color == -1:
            legal = [mv for mv in candidates if not renju_rules.is_forbidden(board, mv[0], mv[1], self.color)]
        if not legal:
            raise ValueError("No legal moves for baseline player")
        return random.choice(legal)


class GreedyBaseline(Player):
    """Greedy baseline: pick legal move with best heuristic score."""

    def __init__(self, color, patterns):
        super().__init__(color)
        self.patterns = patterns

    def next_move(self, board, deadline=None):
        candidates = move_selector.generate_candidates(board, limit=board.size * board.size)
        legal = candidates
        if self.color == -1:
            legal = [mv for mv in candidates if not renju_rules.is_forbidden(board, mv[0], mv[1], self.color)]
        if not legal:
            raise ValueError("No legal moves for baseline player")

        best_score = -10**12
        best_move = legal[0]
        for mv in legal:
            x, y = mv
            board.cells[y][x] = self.color
            board.move_count += 1
            score = heuristic.score_board(board, self.color, patterns=self.patterns)
            board.cells[y][x] = 0
            board.move_count -= 1
            if score > best_score:
                best_score = score
                best_move = mv
        return best_move


def play_game(board_size: int, black, white, timeout: float, random_open: int = 0, epsilon: float = 0.0) -> Tuple[int, List[dict], dict]:
    board = Board(size=board_size)
    to_play = -1
    move_index = 0
    trajectory: List[dict] = []
    timeout_counts = {-1: 0, 1: 0}
    foul_attempts = {-1: 0, 1: 0}
    first_move = None

    while True:
        player = black if to_play == -1 else white
        deadline = time.time() + timeout

        state_board = snapshot_board_state(board)  # capture state before the move
        # Optional randomness to diversify games
        legal_moves = move_selector.generate_candidates(board, limit=board_size * board_size)
        if to_play == -1:
            legal_moves = [mv for mv in legal_moves if not renju_rules.is_forbidden(board, mv[0], mv[1], to_play)]

        move = None
        # Try to get a move from the player
        try:
            # 1. Opening randomness
            if move_index < random_open and legal_moves:
                move = random.choice(legal_moves)
            else:
                # 2. Ask player for move
                move = player.next_move(board, deadline=deadline)
            
            # 3. Epsilon-greedy randomness
            if legal_moves and random.random() < epsilon:
                move = random.choice(legal_moves)

            # 4. Failsafe for fouls (AI might propose forbidden move)
            if renju_rules.is_forbidden(board, move[0], move[1], to_play):
                foul_attempts[to_play] += 1
                # Must pick a legal move
                if not legal_moves:
                    raise ValueError("No legal moves available (all fouls)")
                move = legal_moves[0]  # Deterministic fallback or random.choice(legal_moves)

            # 5. Validate the move (time check included)
            referee.check_move(move, board, to_play, deadline, move_index=move_index)

        except TimeoutError:
            # Handle timeout from player.next_move OR referee.check_move
            # Restore board state in case AI left it dirty
            restore_board_state(board, state_board)
            
            if not legal_moves:
                raise ValueError("Timeout and no legal moves to fallback on")
            timeout_counts[to_play] += 1
            # Fallback to a random legal move
            move = random.choice(legal_moves)
            # We skip referee.check_move for the fallback because we know it comes from legal_moves
            # and we are already handling the timeout penalty by logging it.

        if first_move is None:
            first_move = move

        # store snapshot and policy target
        pi = [0.0] * (board_size * board_size)
        pi[move[1] * board_size + move[0]] = 1.0
        trajectory.append(
            {
                "board": state_board,
                "to_play": to_play,
                "pi": pi,
            }
        )

        board.place(*move, to_play)

        if renju_rules.is_win_after_move(board, *move, to_play):
            winner = to_play
            break
        if board.move_count >= board_size * board_size:
            winner = 0
            break
        move_index += 1
        to_play = -to_play

    # attach outcome from each state's perspective
    for record in trajectory:
        tp = record["to_play"]
        if winner == 0:
            value = 0
        elif winner == tp:
            value = 1
        else:
            value = -1
        record["value"] = value
        record["winner"] = winner
    info = {
        "steps": len(trajectory),
        "timeouts": timeout_counts,
        "first_move": first_move,
        "foul_attempts": foul_attempts,
    }
    # Summarize NPS if stats were collected on players
    nps_summary = {}
    for label, player in [(-1, black), (1, white)]:
        stat_list = getattr(player, "stats", None)
        if stat_list:
            times = [s["time"] for s in stat_list if "time" in s]
            nodes = [s["nodes"] for s in stat_list if "nodes" in s]
            nps_vals = [s["nps"] for s in stat_list if "nps" in s]
            nps_summary[label] = {
                "moves": len(stat_list),
                "time_mean": statistics.mean(times) if times else 0.0,
                "nodes_mean": statistics.mean(nodes) if nodes else 0.0,
                "nps_mean": statistics.mean(nps_vals) if nps_vals else 0.0,
                "nodes_total": sum(nodes) if nodes else 0,
            }
    if nps_summary:
        info["nps"] = nps_summary
    return winner, trajectory, info


def make_baseline_player(color: int, kind: str, patterns):
    if kind == "random":
        return RandomBaseline(color)
    if kind == "greedy":
        return GreedyBaseline(color, patterns)
    return None


def make_players(depth: int, candidate_limit: int, patterns, black_baseline: str, white_baseline: str, pv_helper=None, enable_vcf=False, search_backend="minimax", search_args=None):
    search_args = search_args or {}
    black = make_baseline_player(-1, black_baseline, patterns) or Iot_20203078_KKR(
        color=-1,
        depth=depth,
        candidate_limit=candidate_limit,
        patterns=patterns,
        pv_helper=pv_helper,
        enable_vcf=enable_vcf,
        search_backend=search_backend,
        search_args=search_args.copy(),
    )
    white = make_baseline_player(1, white_baseline, patterns) or Iot_20203078_GIR(
        color=1,
        depth=depth,
        candidate_limit=candidate_limit,
        patterns=patterns,
        pv_helper=pv_helper,
        enable_vcf=enable_vcf,
        search_backend=search_backend,
        search_args=search_args.copy(),
    )
    return black, white


def main():
    parser = argparse.ArgumentParser(description="Renju self-play data generator")
    parser.add_argument("--games", type=int, default=10, help="Number of self-play games to generate")
    parser.add_argument("--board-size", type=int, default=15, help="Board size")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds per move for built-in AIs")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for built-in AIs")
    parser.add_argument("--candidate-limit", type=int, default=15, help="Candidate fan-out for built-in AIs")
    parser.add_argument("--output", default="selfplay_renju.jsonl", help="Output JSONL path")
    parser.add_argument("--swap-colors", action="store_true", help="Swap black/white each game for diversity")
    parser.add_argument("--random-open", type=int, default=0, help="Number of initial plies to choose random legal moves")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Probability of random move each ply (epsilon-greedy)")
    parser.add_argument(
        "--black-baseline",
        choices=["none", "random", "greedy"],
        default="none",
        help="Use a simple baseline for black instead of search AI (evaluation).",
    )
    parser.add_argument(
        "--white-baseline",
        choices=["none", "random", "greedy"],
        default="none",
        help="Use a simple baseline for white instead of search AI (evaluation).",
    )
    parser.add_argument(
        "--collect-stats",
        action="store_true",
        help="Collect per-move stats (nodes/time/nps) for minimax players and summarize at the end.",
    )
    parser.add_argument("--pv-checkpoint", help="Path to policy/value checkpoint for self-play AIs (optional)")
    parser.add_argument("--pv-device", default=None, help="Device for PV model during self-play (cpu or cuda)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for MCTS root noise")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="Dirichlet noise mix fraction at root")
    parser.add_argument("--temperature", type=float, default=1.0, help="Root visit temperature for move sampling")
    parser.add_argument("--search-backend", choices=["minimax", "mcts"], default="minimax", help="Search algorithm to use for built-in AIs")
    parser.add_argument("--enable-vcf", action="store_true", help="Enable VCF search during self-play")
    args = parser.parse_args()

    patterns = heuristic.load_patterns()
    # Optional PV helper for self-play
    pv_path = Path(args.pv_checkpoint) if args.pv_checkpoint else Path("checkpoints/pv_latest.pt")
    pv_device = args.pv_device or "cpu"
    pv_helper = None
    if pv_path.exists():
        pv_helper = policy_value.PolicyValueInfer(str(pv_path), device=pv_device)
        print(f"Loaded PV checkpoint for self-play: {pv_path} (device={pv_device})")
    else:
        if args.pv_checkpoint:
            print(f"Warning: PV checkpoint not found at {pv_path}, proceeding without PV model.")
    
    # Build search args for MCTS (only used if backend == mcts)
    search_args = {
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_frac": args.dirichlet_frac,
        "temperature": args.temperature,
        "candidate_limit": args.candidate_limit,
    }

    black, white = make_players(
        args.depth, 
        args.candidate_limit, 
        patterns, 
        args.black_baseline, 
        args.white_baseline, 
        pv_helper=pv_helper, 
        enable_vcf=args.enable_vcf,
        search_backend=args.search_backend,
        search_args=search_args,
    )
    if args.collect_stats:
        setattr(black, "stats", [])
        setattr(white, "stats", [])

    count = 0
    lengths = []
    winners = Counter()
    timeouts = Counter()
    openings = Counter()
    fouls = Counter()
    nps_logs = []
    with open(args.output, "w", encoding="utf-8") as f:
        for g in range(args.games):
            # Swap colors every other game if enabled
            b_player, w_player = (black, white) if not args.swap_colors or g % 2 == 0 else (white, black)
            winner, traj, info = play_game(
                args.board_size,
                b_player,
                w_player,
                args.timeout,
                random_open=args.random_open,
                epsilon=args.epsilon,
            )
            for rec in traj:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
            count += 1
            lengths.append(info["steps"])
            winners[winner] += 1
            for color, c in info["timeouts"].items():
                if c:
                    timeouts[color] += c
            if info["first_move"] is not None:
                openings[info["first_move"]] += 1
            for color, c in info["foul_attempts"].items():
                if c:
                    fouls[color] += c
            if "nps" in info:
                nps_logs.append(info["nps"])
            print(
                f"[{g+1}/{args.games}] winner={winner}, steps={info['steps']}, "
                f"timeouts={info['timeouts']}, foul_attempts={info['foul_attempts']}"
            )

    if lengths:
        mean_len = statistics.mean(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Steps: mean={mean_len:.1f}, min={min_len}, max={max_len}")
    print(f"Winners: {dict(winners)}")
    if timeouts:
        print(f"Timeouts: {dict(timeouts)} (fallback-to-random counts)")
    if openings:
        top_openings = openings.most_common(5)
        print(f"Top openings (x,y,count): {[(x, y, c) for ((x, y), c) in top_openings]}")
    if fouls:
        print(f"Foul attempts (blocked forbidden moves): {dict(fouls)}")
    if nps_logs:
        agg = {-1: [], 1: []}
        for entry in nps_logs:
            for color in (-1, 1):
                if color in entry:
                    agg[color].append(entry[color])
        for color, items in agg.items():
            if items:
                moves = sum(d["moves"] for d in items)
                time_mean = statistics.mean(d["time_mean"] for d in items)
                nps_mean = statistics.mean(d["nps_mean"] for d in items)
                print(f"NPS color {color}: moves={moves}, time_mean={time_mean:.3f}s, nps_mean={nps_mean:.0f}")
    
    print(f"Saved {count} games to {args.output}")

    # Save summary stats to JSON for auto_train logging
    stats_path = Path(args.output).with_name(Path(args.output).stem + "_stats.json")
    summary = {
        "games": count,
        "black_wins": winners[-1],
        "white_wins": winners[1],
        "draws": winners[0],
        "avg_steps": mean_len if lengths else 0,
        "black_timeouts": timeouts[-1],
        "white_timeouts": timeouts[1],
        "black_fouls": fouls[-1],
        "white_fouls": fouls[1],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
