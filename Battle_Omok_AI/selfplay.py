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

from Board import Board
from engine import referee, renju_rules
from Iot_20203078_KKR import Iot_20203078_KKR
from Iot_20203078_GIR import Iot_20203078_GIR
from Player import Player
from ai import move_selector, heuristic, policy_value, search_minimax, search_mcts


def snapshot(board: Board) -> List[List[int]]:
    return [row[:] for row in board.cells]


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

        state_board = snapshot(board)  # capture state before the move
        # Optional randomness to diversify games
        legal_moves = move_selector.generate_candidates(board, limit=board_size * board_size)
        if to_play == -1:
            legal_moves = [mv for mv in legal_moves if not renju_rules.is_forbidden(board, mv[0], mv[1], to_play)]

        move = None
        if move_index < random_open and legal_moves:
            move = random.choice(legal_moves)
        else:
            try:
                move = player.next_move(board, deadline=deadline)
            except TimeoutError:
                # Fallback to legal random move on timeout to keep self-play going.
                if not legal_moves:
                    raise
                timeout_counts[to_play] += 1
                move = random.choice(legal_moves)
            if renju_rules.is_forbidden(board, move[0], move[1], to_play) and legal_moves:
                move = legal_moves[0]

        # epsilon-greedy randomization
        if legal_moves and random.random() < epsilon:
            move = random.choice(legal_moves)

        # Failsafe: if AI proposes a forbidden move (e.g., due to stale cache), pick the first legal candidate.
        if renju_rules.is_forbidden(board, move[0], move[1], to_play):
            foul_attempts[to_play] += 1
            candidates = move_selector.generate_candidates(board, limit=board_size * board_size)
            legal = [mv for mv in candidates if not renju_rules.is_forbidden(board, mv[0], mv[1], to_play)]
            if not legal:
                raise ValueError("No legal moves available for black (Renju fouls everywhere)")
            move = legal[0]

        if first_move is None:
            first_move = move

        # store snapshot and policy target from the perspective of the current player (pre-move)
        pi = [0.0] * (board_size * board_size)
        pi[move[1] * board_size + move[0]] = 1.0
        trajectory.append(
            {
                "board": state_board,
                "to_play": to_play,
                "pi": pi,
            }
        )

        try:
            referee.check_move(move, board, to_play, deadline, move_index=move_index)
        except TimeoutError:
            # If we hit the deadline before validation, fall back to a fast legal move and re-check.
            timeout_counts[to_play] += 1
            if legal_moves:
                move = legal_moves[0]
            new_deadline = time.time() + 0.5  # short grace to pass validation
            referee.check_move(move, board, to_play, new_deadline, move_index=move_index)
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


def make_players(depth: int, candidate_limit: int, patterns, black_baseline: str, white_baseline: str):
    black = make_baseline_player(-1, black_baseline, patterns) or Iot_20203078_KKR(
        color=-1, depth=depth, candidate_limit=candidate_limit, patterns=patterns
    )
    white = make_baseline_player(1, white_baseline, patterns) or Iot_20203078_GIR(
        color=1, depth=depth, candidate_limit=candidate_limit, patterns=patterns
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
    args = parser.parse_args()

    patterns = heuristic.load_patterns()
    # Optional PV helper for self-play
    pv_path = Path(args.pv_checkpoint) if args.pv_checkpoint else Path("checkpoints/pv_latest.pt")
    pv_device = args.pv_device or "cpu"
    if pv_path.exists():
        pv_helper = policy_value.PolicyValueInfer(str(pv_path), device=pv_device)
        search_minimax.PV_HELPER = pv_helper
        search_mcts.PV_HELPER = pv_helper
        print(f"Loaded PV checkpoint for self-play: {pv_path} (device={pv_device})")
    else:
        if args.pv_checkpoint:
            print(f"Warning: PV checkpoint not found at {pv_path}, proceeding without PV model.")
    black, white = make_players(args.depth, args.candidate_limit, patterns, args.black_baseline, args.white_baseline)
    if args.collect_stats:
        setattr(black, "stats", [])
        setattr(white, "stats", [])
    # Pass exploration knobs to MCTS (if used)
    for p in (black, white):
        if hasattr(p, "search_args"):
            p.search_args.update(
                {
                    "dirichlet_alpha": args.dirichlet_alpha,
                    "dirichlet_frac": args.dirichlet_frac,
                    "temperature": args.temperature,
                }
            )

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


if __name__ == "__main__":
    main()
