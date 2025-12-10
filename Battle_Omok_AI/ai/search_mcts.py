"""Lightweight MCTS with random/bias-rollouts to provide a usable fallback search."""

import math
import random
import time

from . import move_selector
try:
    from engine import renju_rules
except ImportError:
    from Battle_Omok_AI.engine import renju_rules


class _Node:
    __slots__ = ("move", "parent", "children", "untried", "visits", "reward")

    def __init__(self, move=None, parent=None, untried=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.untried = untried or []
        self.visits = 0
        self.reward = 0.0

    def ucb1(self, explore=1.4):
        if self.visits == 0:
            return float("inf")
        return (self.reward / self.visits) + explore * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


def choose_move(board, color, deadline, rollout_limit=512, candidate_limit=24, explore=1.4):
    """
    Return a move using Monte Carlo Tree Search with simple rollouts.
    - color: player to move (-1 or 1)
    - deadline: epoch seconds to stop
    """
    def filtered_candidates(b, c):
        cands = move_selector.generate_candidates(b, limit=candidate_limit)
        if c == -1:
            cands = [mv for mv in cands if not renju_rules.is_forbidden(b, mv[0], mv[1], c)]
        return cands

    candidates = filtered_candidates(board, color)
    if not candidates:
        for y in range(board.size):
            for x in range(board.size):
                if board.is_empty(x, y) and (color != -1 or not renju_rules.is_forbidden(board, x, y, color)):
                    return (x, y)
        return (0, 0)

    root = _Node(move=None, parent=None, untried=list(candidates))
    rollout_count = 0

    def time_ok():
        if time.time() > deadline:
            raise TimeoutError("MCTS timed out")

    def rollout(sim_board, to_move):
        """Random rollout biased by nearby-candidate selection."""
        while True:
            wins = filtered_candidates(sim_board, to_move)
            if not wins:
                return 0  # draw
            move = random.choice(wins)
            sim_board.place(*move, to_move)
            if renju_rules.is_win_after_move(sim_board, *move, to_move):
                return 1 if to_move == color else -1
            if sim_board.move_count >= sim_board.size * sim_board.size:
                return 0
            to_move = -to_move

    try:
        while rollout_count < rollout_limit:
            time_ok()
            node = root
            sim_board = board.clone()
            to_move = color

            # Selection
            while not node.untried and node.children:
                node = max(node.children, key=lambda n: n.ucb1(explore))
                sim_board.place(*node.move, to_move)
                if sim_board.has_exact_five(*node.move):
                    result = 1 if to_move == color else -1
                    backprop(node, result)
                    rollout_count += 1
                    break
                to_move = -to_move
            else:
                # Expansion
                if node.untried:
                    move = node.untried.pop()
                    sim_board.place(*move, to_move)
                    next_color = -to_move
                    child = _Node(move=move, parent=node, untried=filtered_candidates(sim_board, next_color))
                    node.children.append(child)
                    node = child
                    to_move = -to_move

                # Simulation
                result = rollout(sim_board, to_move)
                rollout_count += 1
                backprop(node, result)
    except TimeoutError:
        pass

    # Pick the child with the most visits (robust child).
    if not root.children:
        return candidates[0]
    best = max(root.children, key=lambda n: n.visits)
    return best.move


def backprop(node, result):
    while node is not None:
        node.visits += 1
        node.reward += result
        node = node.parent
