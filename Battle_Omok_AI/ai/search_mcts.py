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
from typing import List, Optional, Tuple

from . import move_selector
try:
    from engine import renju_rules
except ImportError:
    from Battle_Omok_AI.engine import renju_rules

# Optional PV helper (ai.policy_value.PolicyValueInfer). Set externally.
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
        """PUCT score; assumes value_sum is stored from this node's to-move perspective (no sign flips here)."""
        q = 0.0 if self.visits == 0 else self.value_sum / self.visits
        u = c_puct * self.prior * math.sqrt(parent_visits + 1e-8) / (1 + self.visits)
        return q + u


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
):
    """
    Return a move using PUCT MCTS guided by optional policy/value helper.
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

    # Root priors (with optional Dirichlet noise)
    root_priors = {}
    if PV_HELPER is not None:
        probs, _ = PV_HELPER.predict(board.cells, color)
        for mv in candidates:
            idx = mv[1] * board.size + mv[0]
            root_priors[mv] = probs[idx].item()
    else:
        uniform = 1.0 / max(len(candidates), 1)
        for mv in candidates:
            root_priors[mv] = uniform

    if dirichlet_alpha > 0 and dirichlet_frac > 0:
        import torch

        noise = torch.distributions.Dirichlet(torch.full((len(candidates),), dirichlet_alpha)).sample()
        for i, mv in enumerate(candidates):
            root_priors[mv] = (1 - dirichlet_frac) * root_priors[mv] + dirichlet_frac * noise[i].item()

    root = _Node(move=None, parent=None, untried=list(candidates), prior=1.0)

    def time_ok():
        if time.time() > deadline:
            raise TimeoutError("MCTS timed out")

    def expand_and_eval(node: _Node, sim_board, to_move: int) -> float:
        """Compute priors/value once for this node, add placeholder children, and return value (current player perspective)."""
        if node.priors is None:
            moves = node.untried or filtered_candidates(sim_board, to_move)
            node.untried = list(moves)
            priors = {}
            val = 0.0
            if PV_HELPER is not None:
                probs, val = PV_HELPER.predict(sim_board.cells, to_move)
                for mv in moves:
                    idx = mv[1] * sim_board.size + mv[0]
                    priors[mv] = probs[idx].item()
            else:
                priors = {mv: 1.0 / max(len(moves), 1) for mv in moves}
                val = 0.0
            node.priors = priors
            node.value_estimate = float(val) if PV_HELPER is not None else 0.0

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
            sim_board = board.clone()
            to_move = color

            # Selection
            while node.is_expanded and node.children:
                node = max(node.children, key=lambda n: n.puct(node.visits, explore))
                sim_board.place(*node.move, to_move)
                if renju_rules.is_win_after_move(sim_board, *node.move, to_move):
                    # Current player wins; pass +1 (current perspective). backprop flips for parent.
                    node.terminal = True
                    node.is_expanded = True
                    node.children = []
                    node.untried = []
                    backprop(node, 1.0)
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
                    backprop(node, 0.0)
                    rollout_count += 1
                    continue
    except TimeoutError:
        pass

    # Pick move
    if not root.children:
        return candidates[0]
    if temperature <= 1e-3:
        best = max(root.children, key=lambda n: n.visits)
        return best.move
    import math as _math
    visits = [c.visits for c in root.children]
    max_v = max(visits)
    probs = []
    for v in visits:
        probs.append((v + 1e-8) ** (1.0 / temperature))
    total = sum(probs)
    r = (_math.fsum(probs) * 0)  # dummy to keep fsum import warning quiet
    # simple multinomial sampling without numpy
    rnd = (total) * (time.time() % 1.0)
    acc = 0.0
    for child, p in zip(root.children, probs):
        acc += p
        if rnd <= acc:
            return child.move
    return root.children[-1].move
