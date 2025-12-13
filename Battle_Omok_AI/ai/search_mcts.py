"""PUCT MCTS for two-player zero-sum Renju/Gomoku with optional PV helper.

Key design:
- Negamax backup: values are from the perspective of the player to move at each node;
  during backprop we flip the sign each ply.
- No rollouts: leaf values come from the value head (if available), else 0.
- Priors are injected once per expansion; PUCT selection uses those priors.
- Black forbidden moves are filtered via renju_rules.
"""

from __future__ import annotations

import math
import time
import random
from typing import List, Optional, Tuple

from . import move_selector
try:
    from engine import renju_rules
except ImportError:
    from Battle_Omok_AI.engine import renju_rules

# Optional global PV helper for backward compatibility
PV_HELPER = None

class _Node:
    __slots__ = (
        "move",
        "parent",
        "children",
        "untried",
        "visits",
        "value_sum",
        "prior",
        "is_expanded",
        "priors",
        "value_estimate",
        "terminal",
    )

    def __init__(
        self,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["_Node"] = None,
        untried: Optional[List[Tuple[int, int]]] = None,
        prior: float = 0.0,
    ):
        self.move = move
        self.parent = parent
        self.children: List[_Node] = []
        self.untried = list(untried or [])
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.priors = None  # move -> prior
        self.value_estimate = 0.0
        self.terminal = False  # whether this node is a terminal state

    def puct(self, parent_visits: int, c_puct: float) -> float:
        """
        PUCT score from the *parent* perspective.

        value_sum is stored from this node's to-move perspective, so the parent
        sees the negated Q when choosing among children.
        """
        q_self = 0.0 if self.visits == 0 else self.value_sum / self.visits
        q_parent = -q_self
        u = c_puct * self.prior * math.sqrt(parent_visits + 1e-8) / (1 + self.visits)
        return q_parent + u


def choose_move(
    board,
    color: int,
    deadline,
    rollout_limit: int = 512,
    candidate_limit: int = 24,
    explore: float = 1.4,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
    temperature: float = 1.0,
    pv_helper=None,
    return_pi: bool = False,
):
    """
    Return a move using PUCT MCTS guided by optional policy/value helper.
    - color: player to move (-1 or 1)
    - deadline: epoch seconds to stop
    - return_pi: if True, also return root visit distribution (pi) as a list
    """

    pv_helper = pv_helper or PV_HELPER

    def filtered_candidates(b, c):
        cands = move_selector.generate_candidates(b, limit=candidate_limit)
        if c == -1:
            cands = [mv for mv in cands if not renju_rules.is_forbidden(b, mv[0], mv[1], c)]
            if not cands:
                # Rare fallback: if all local candidates are forbidden, allow any legal empty.
                legal_all = [
                    (x, y)
                    for y in range(b.size)
                    for x in range(b.size)
                    if b.is_empty(x, y) and not renju_rules.is_forbidden(b, x, y, c)
                ]
                cands = legal_all[:candidate_limit]
        return cands

    candidates = filtered_candidates(board, color)
    if not candidates:
        legal_all: List[Tuple[int, int]] = []
        for y in range(board.size):
            for x in range(board.size):
                if not board.is_empty(x, y):
                    continue
                if color == -1 and renju_rules.is_forbidden(board, x, y, color):
                    continue
                legal_all.append((x, y))
        if not legal_all:
            raise ValueError("No legal moves available for MCTS")

        move = legal_all[0]
        if return_pi:
            pi = [0.0] * (board.size * board.size)
            pi[move[1] * board.size + move[0]] = 1.0
            return move, pi
        return move

    # Fast path: if there is an immediate winning move, play it directly.
    for mv in candidates:
        x, y = mv
        board._push_stone(x, y, color)
        try:
            if renju_rules.is_win_after_move(board, x, y, color):
                if return_pi:
                    pi = [0.0] * (board.size * board.size)
                    pi[y * board.size + x] = 1.0
                    return mv, pi
                return mv
        finally:
            board._pop_stone(x, y)

    # Full-board tactical guardrails:
    # - If a winning move exists but got truncated out of candidates, still play it.
    # - If the opponent has an immediate winning move, try to block it directly.
    for y in range(board.size):
        for x in range(board.size):
            if not board.is_empty(x, y):
                continue
            board._push_stone(x, y, color)
            try:
                if renju_rules.is_win_after_move(board, x, y, color):
                    mv = (x, y)
                    if return_pi:
                        pi = [0.0] * (board.size * board.size)
                        pi[y * board.size + x] = 1.0
                        return mv, pi
                    return mv
            finally:
                board._pop_stone(x, y)

    opp = -color
    threat_moves: List[Tuple[int, int]] = []
    for y in range(board.size):
        for x in range(board.size):
            if not board.is_empty(x, y):
                continue
            board._push_stone(x, y, opp)
            try:
                if renju_rules.is_win_after_move(board, x, y, opp):
                    threat_moves.append((x, y))
            finally:
                board._pop_stone(x, y)

    if threat_moves:
        for x, y in threat_moves:
            if not board.is_empty(x, y):
                continue
            if color == -1 and renju_rules.is_forbidden(board, x, y, color):
                continue
            mv = (x, y)
            if return_pi:
                pi = [0.0] * (board.size * board.size)
                pi[y * board.size + x] = 1.0
                return mv, pi
            return mv

    def normalize_priors(priors: dict, moves: List[Tuple[int, int]]):
        total = sum(priors.get(mv, 0.0) for mv in moves)
        if total <= 0:
            uniform = 1.0 / max(len(moves), 1)
            for mv in moves:
                priors[mv] = uniform
            return
        for mv in moves:
            priors[mv] = priors.get(mv, 0.0) / total

    # Root priors (with optional Dirichlet noise)
    root_priors: dict[Tuple[int, int], float] = {}
    root_value = 0.0
    if pv_helper is not None:
        probs, root_value = pv_helper.predict(board.cells, color)
        for mv in candidates:
            idx = mv[1] * board.size + mv[0]
            root_priors[mv] = probs[idx].item()
    else:
        uniform = 1.0 / max(len(candidates), 1)
        for mv in candidates:
            root_priors[mv] = uniform

    # Normalize and inject Dirichlet noise at root for exploration (useful for self-play).
    normalize_priors(root_priors, candidates)
    if dirichlet_alpha > 0 and dirichlet_frac > 0 and len(candidates) > 1:
        try:
            import torch

            noise = torch.distributions.Dirichlet(
                torch.full((len(candidates),), dirichlet_alpha)
            ).sample()
            for i, mv in enumerate(candidates):
                root_priors[mv] = (1 - dirichlet_frac) * root_priors[mv] + dirichlet_frac * noise[i].item()
            normalize_priors(root_priors, candidates)
        except ImportError:
            pass

    root = _Node(move=None, parent=None, untried=[], prior=1.0)
    root.priors = root_priors
    # Explicitly populate root children so Selection can pick among them immediately
    for mv in candidates:
        child = _Node(move=mv, parent=root, untried=[], prior=root_priors.get(mv, 0.0))
        root.children.append(child)
    root.is_expanded = True
    root.value_estimate = float(root_value) if pv_helper is not None else 0.0

    def time_ok():
        if time.time() > deadline:
            raise TimeoutError("MCTS timed out")

    def expand_and_eval(node: _Node, sim_board, to_move: int) -> float:
        """Compute priors/value once for this node, add placeholder children, and return value (current player perspective)."""
        if node.priors is None:
            moves = node.untried or filtered_candidates(sim_board, to_move)
            node.untried = list(moves)
            if not moves:
                # No legal moves for side to move -> loss from this node's perspective.
                node.priors = {}
                node.children = []
                node.untried = []
                node.terminal = True
                node.value_estimate = -1.0
                node.is_expanded = True
                return node.value_estimate
            priors = {}
            val = 0.0
            if pv_helper is not None:
                probs, val = pv_helper.predict(sim_board.cells, to_move)
                for mv in moves:
                    idx = mv[1] * sim_board.size + mv[0]
                    priors[mv] = probs[idx].item()
            else:
                priors = {mv: 1.0 / max(len(moves), 1) for mv in moves}
                val = 0.0
            normalize_priors(priors, moves)
            node.priors = priors
            node.value_estimate = float(val) if pv_helper is not None else 0.0

            if not node.children:
                for mv in moves:
                    node.children.append(
                        _Node(
                            move=mv,
                            parent=node,
                            untried=[],
                            prior=node.priors.get(mv, 0.0),
                        )
                    )
            node.untried = []
        node.is_expanded = True

        return node.value_estimate

    def backprop(node: _Node, value: float):
        """Negamax backup: value is for current node's to-move; flip after updating for parent."""
        cur = node
        v = value
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = -v  # flip for parent on next step
            cur = cur.parent

    rollout_count = 0
    try:
        while rollout_count < rollout_limit:
            time_ok()
            node = root
            sim_board = board
            to_move = color
            applied_moves: List[Tuple[int, int]] = []

            try:
                # Selection
                while node.is_expanded and node.children:
                    node = max(node.children, key=lambda n: n.puct(node.visits, explore))
                    x, y = node.move
                    sim_board._push_stone(x, y, to_move)
                    applied_moves.append((x, y))

                    if renju_rules.is_win_after_move(sim_board, x, y, to_move):
                        # Previous player (to_move) has just won, so from this node's
                        # to-move (opponent) perspective the value is -1.
                        node.terminal = True
                        node.is_expanded = True
                        node.children = []
                        node.untried = []
                        backprop(node, -1.0)
                        rollout_count += 1
                        break
                    if sim_board.move_count >= sim_board.size * sim_board.size:
                        node.terminal = True
                        node.is_expanded = True
                        node.children = []
                        node.untried = []
                        backprop(node, 0.0)
                        rollout_count += 1
                        break
                    to_move = -to_move
                else:
                    # Expansion/Eval
                    if not node.is_expanded:
                        value = expand_and_eval(node, sim_board, to_move)
                        backprop(node, value)
                        rollout_count += 1
                        continue
                    elif not node.children:
                        # No legal moves at this node -> loss for side to move.
                        backprop(node, -1.0)
                        rollout_count += 1
                        continue
            finally:
                # Undo simulation moves to restore original board.
                for x, y in reversed(applied_moves):
                    sim_board._pop_stone(x, y)
    except TimeoutError:
        pass

    def build_pi_from_children(children: List[_Node]) -> list[float]:
        size = board.size
        pi = [0.0] * (size * size)
        if not children:
            return pi
        if temperature <= 1e-3:
            best_child = max(children, key=lambda n: (n.visits, random.random()))
            idx = best_child.move[1] * size + best_child.move[0]
            pi[idx] = 1.0
            return pi

        visits = [c.visits for c in children]
        weights = [(v + 1e-8) ** (1.0 / temperature) for v in visits]
        total_w = sum(weights)
        if total_w <= 0:
            uniform = 1.0 / max(len(children), 1)
            for c in children:
                idx = c.move[1] * size + c.move[0]
                pi[idx] = uniform
            return pi

        for c, w in zip(children, weights):
            idx = c.move[1] * size + c.move[0]
            pi[idx] = w / total_w
        return pi

    # Pick move and optional pi
    if not root.children:
        move = candidates[0]
        if return_pi:
            pi = [0.0] * (board.size * board.size)
            pi[move[1] * board.size + move[0]] = 1.0
            return move, pi
        return move

    if temperature <= 1e-3:
        best = max(root.children, key=lambda n: (n.visits, random.random()))
        move = best.move
        if return_pi:
            pi = build_pi_from_children(root.children)
            return move, pi
        return move

    visits = [c.visits for c in root.children]
    weights = [(v + 1e-8) ** (1.0 / temperature) for v in visits]
    total = sum(weights)
    if total <= 0:
        move = random.choice(root.children).move
        if return_pi:
            pi = build_pi_from_children(root.children)
            return move, pi
        return move

    rnd = random.random() * total
    acc = 0.0
    move = root.children[-1].move
    for child, w in zip(root.children, weights):
        acc += w
        if rnd <= acc:
            move = child.move
            break

    if return_pi:
        pi = build_pi_from_children(root.children)
        return move, pi
    return move
