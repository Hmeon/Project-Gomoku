"""Minimax with alpha-beta pruning, VCF probe, and optimized ordering under time budget."""

import time

from . import heuristic
from . import move_selector
from . import transposition
try:
    from engine import renju_rules
except ImportError:
    from Battle_Omok_AI.engine import renju_rules

# Optional global PV helper (ai.policy_value.PolicyValueInfer); can be set externally.
PV_HELPER = None
PV_SCALE = 5000  # scale factor to mix PV value into heuristic

INF = 10 ** 9
TIME_CHECK_MASK = 2047  # check every 2048 nodes


def choose_move(board, color, depth, deadline, cache=None, zobrist_table=None, candidate_limit=15, patterns=None, pv_helper=None, stats=None):
    """
    Return best move for color within depth and before deadline using iterative deepening.
    - board: Board instance
    - color: -1 (black) or 1 (white)
    """
    if zobrist_table is None:
        zobrist_table = transposition.zobrist_init(board.size)
    if cache is None:
        cache = {}
    patterns = patterns or heuristic.DEFAULT_PATTERNS
    pv_helper = pv_helper or PV_HELPER

    # First try a narrow VCF search for forced wins.
    vcf_move = _solve_vcf(board, color, deadline, candidate_limit=min(candidate_limit, 12))
    if vcf_move is not None:
        return vcf_move

    node_counter = 0
    node_box = [0]
    start_time = time.time()
    best_move = None
    pv_move = None

    def time_ok():
        nonlocal node_counter
        node_counter += 1
        node_box[0] += 1
        if node_counter & TIME_CHECK_MASK == 0:
            if time.time() > deadline:
                raise TimeoutError("Search timed out")

    def minimax(node_color, d, alpha, beta):
        nonlocal pv_move
        time_ok()

        # Terminal check: draw or depth 0
        if d == 0 or board.move_count == board.size * board.size:
            score = heuristic.score_board(board, color, patterns=patterns)
            if pv_helper is not None:
                _, val = pv_helper.predict(board.cells, node_color)
                val_score = val if node_color == color else -val
                score += int(val_score * PV_SCALE)
            return score, None

        key = (transposition.hash_board(board, zobrist_table), node_color)
        cached = cache.get(key)
        if cached:
            cached_depth, cached_score, cached_flag, cached_move = cached
            if cached_depth >= d:
                if cached_flag == "EXACT":
                    return cached_score, cached_move
                if cached_flag == "LOWER":
                    alpha = max(alpha, cached_score)
                elif cached_flag == "UPPER":
                    beta = min(beta, cached_score)
                if alpha >= beta:
                    return cached_score, cached_move

        maximizing = (node_color == color)
        best_score = -float("inf") if maximizing else float("inf")
        best_local_move = None
        alpha_orig = alpha

        candidates = move_selector.generate_candidates(board, limit=candidate_limit)
        if not candidates:
            return heuristic.score_board(board, color, patterns=patterns), None

        pv_probs = None
        if pv_helper is not None and d == depth:
            probs, _ = pv_helper.predict(board.cells, node_color)
            pv_probs = probs

        ordered = _order_moves(
            board,
            candidates,
            node_color,
            patterns,
            pv_hint=pv_move if d == depth else None,
            pv_probs=pv_probs,
        )

        for move, win_now, _ in ordered:
            x, y = move
            # Lazy forbidden check: only when the move is about to be expanded
            if node_color == -1 and renju_rules.is_forbidden(board, x, y, node_color):
                continue
            board.cells[y][x] = node_color
            board.move_count += 1

            if win_now:
                score = (INF - board.move_count) if maximizing else (-INF + board.move_count)
            else:
                score, _ = minimax(-node_color, d - 1, alpha, beta)

            # undo
            board.cells[y][x] = 0
            board.move_count -= 1

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_local_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_local_move = move
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        flag = "EXACT"
        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        cache[key] = (d, best_score, flag, best_local_move)
        return best_score, best_local_move

    for current_depth in range(1, depth + 1):
        try:
            score, move = minimax(color, current_depth, -float("inf"), float("inf"))
            if move is not None:
                best_move = move
                pv_move = move  # principal variation move searched first next iter
        except TimeoutError:
            break

    if best_move is None:
        # fallback: first empty
        for y in range(board.size):
            for x in range(board.size):
                if board.is_empty(x, y):
                    if color == -1 and renju_rules.is_forbidden(board, x, y, color):
                        continue
                    best_move = (x, y)
                    break
            if best_move:
                break

    if stats is not None:
        total_time = max(time.time() - start_time, 1e-9)
        stats.append(
            {
                "color": color,
                "depth": depth,
                "nodes": node_box[0],
                "time": total_time,
                "nps": node_box[0] / total_time,
            }
        )

    return best_move


def _order_moves(board, candidates, node_color, patterns, pv_hint=None, pv_probs=None):
    """Lightweight move ordering: PV hint first, PV prior (if any), immediate wins, then local density."""
    ordered = []
    opp = -node_color
    for move in candidates:
        x, y = move
        if board.cells[y][x] != 0:
            continue
        board.cells[y][x] = node_color
        board.move_count += 1
        win_now = renju_rules.is_win_after_move(board, x, y, node_color)
        local_score = INF // 2 if win_now else _local_density(board, x, y, node_color, opp)
        board.cells[y][x] = 0
        board.move_count -= 1
        if pv_hint and move == pv_hint:
            local_score += INF // 4
        if pv_probs is not None:
            idx = y * board.size + x
            local_score += pv_probs[idx].item() * (PV_SCALE / 2)
        ordered.append((move, win_now, local_score))

    ordered.sort(key=lambda item: item[2], reverse=True)
    return ordered


def _local_density(board, x, y, node_color, opp):
    """Heuristic based on nearby stones; avoids full board evaluation."""
    score = 0
    for dy in (-2, -1, 0, 1, 2):
        ny = y + dy
        if ny < 0 or ny >= board.size:
            continue
        for dx in (-2, -1, 0, 1, 2):
            nx = x + dx
            if nx < 0 or nx >= board.size or (dx == 0 and dy == 0):
                continue
            val = board.cells[ny][nx]
            if val == node_color:
                score += 6 if abs(dx) <= 1 and abs(dy) <= 1 else 3
            elif val == opp:
                score += 4 if abs(dx) <= 1 and abs(dy) <= 1 else 2
            else:
                score += 1
    # favor center slightly
    center = board.size // 2
    score -= abs(x - center) + abs(y - center)
    return score


def _solve_vcf(board, color, deadline, candidate_limit=8, max_depth=4):
    """Very narrow VCF-style search: look for double threats or immediate wins.
    Uses a smaller candidate_limit to reduce branching and probe deeper."""
    def time_guard():
        if time.time() > deadline:
            raise TimeoutError

    def immediate_wins(turn):
        wins = []
        cands = move_selector.generate_candidates(board, limit=candidate_limit)
        for mv in cands:
            x, y = mv
            if board.cells[y][x] != 0:
                continue
            if turn == -1 and renju_rules.is_forbidden(board, x, y, turn):
                continue
            board.cells[y][x] = turn
            board.move_count += 1
            if renju_rules.is_win_after_move(board, x, y, turn):
                wins.append(mv)
            board.cells[y][x] = 0
            board.move_count -= 1
        return wins

    def dfs(turn, depth_left):
        time_guard()
        win_moves = immediate_wins(turn)
        if win_moves:
            # Multiple threats are effectively forced
            return win_moves[0] if len(win_moves) >= 2 else win_moves[0]
        if depth_left == 0:
            return None

        cands = move_selector.generate_candidates(board, limit=candidate_limit)
        for mv in cands:
            x, y = mv
            if board.cells[y][x] != 0:
                continue
            if turn == -1 and renju_rules.is_forbidden(board, x, y, turn):
                continue
            board.cells[y][x] = turn
            board.move_count += 1
            created_wins = immediate_wins(turn)
            forced = False
            if len(created_wins) >= 2:
                forced = True
            elif len(created_wins) == 1 and depth_left > 0:
                block = created_wins[0]
                bx, by = block
                board.cells[by][bx] = -turn
                board.move_count += 1
                forced = dfs(turn, depth_left - 1) is not None
                board.cells[by][bx] = 0
                board.move_count -= 1
            board.cells[y][x] = 0
            board.move_count -= 1
            if forced:
                return mv
        return None

    try:
        return dfs(color, max_depth)
    except TimeoutError:
        return None
