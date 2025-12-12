"""Minimax with alpha-beta pruning, VCF probe, and optimized ordering under time budget."""

import time
import random

from . import heuristic
from . import move_selector
from . import transposition

try:
    from engine import renju_rules
except ImportError:
    from Battle_Omok_AI.engine import renju_rules


INF = 10 ** 9
PV_SCALE = 5000  # scale factor to mix PV value into heuristic
TIME_CHECK_MASK = 2047  # check every 2048 nodes


class MinimaxSearcher:
    """Encapsulates the state and logic for a minimax search."""

    def __init__(self, board_size, color, depth, candidate_limit, patterns, pv_helper=None, cache=None, zobrist_table=None, stats=None, enable_vcf=False):
        self.board_size = board_size
        self.color = color
        self.depth = depth
        self.candidate_limit = candidate_limit
        self.patterns = patterns or heuristic.DEFAULT_PATTERNS
        self.pv_helper = pv_helper
        self.cache = {} if cache is None else cache
        self.zobrist_table = transposition.zobrist_init(board_size) if zobrist_table is None else zobrist_table
        self.stats_list = stats
        self.enable_vcf = enable_vcf
        
        # Internal state
        self.node_counter = 0
        self.deadline = None
        self.start_time = None
        self.pv_move = None
        self.root_score = None

    def choose_move(self, board, deadline):
        """
        Return best move for color within depth and before deadline using iterative deepening.
        """
        self.deadline = deadline
        self.start_time = time.time()
        self.root_score = heuristic.score_board(board, self.color, patterns=self.patterns)
        
        # First try a narrow VCF search for forced wins, if enabled.
        if self.enable_vcf:
            vcf_move = self._solve_vcf(board, self.color, candidate_limit=min(self.candidate_limit, 12))
            if vcf_move is not None:
                return vcf_move

        best_move = None
        for current_depth in range(1, self.depth + 1):
            try:
                _, move = self._minimax(board, self.color, current_depth, -float("inf"), float("inf"), self.root_score, last_move=None)
                if move is not None:
                    best_move = move
                    self.pv_move = move  # Principal variation move for next iteration
            except TimeoutError:
                break
        
        # Fallback if no move is found
        if best_move is None:
            best_move = self._fallback_move(board)

        if self.stats_list is not None:
            self._record_stats()

        return best_move

    def _time_ok(self):
        self.node_counter += 1
        if (self.node_counter & TIME_CHECK_MASK) == 0:
            if time.time() > self.deadline:
                raise TimeoutError("Search timed out")

    def _minimax(self, board, node_color, depth, alpha, beta, current_score, last_move):
        self._time_ok()

        # Terminal state check
        if depth == 0 or board.move_count == self.board_size * self.board_size:
            return self._evaluate_terminal(board, node_color, current_score), None

        # Transposition table lookup
        key = (transposition.hash_board(board, self.zobrist_table), node_color)
        cached = self.cache.get(key)
        if cached:
            cached_depth, cached_score, cached_flag, cached_move = cached
            if cached_depth >= depth:
                if cached_flag == "EXACT":
                    return cached_score, cached_move
                if cached_flag == "LOWER":
                    alpha = max(alpha, cached_score)
                elif cached_flag == "UPPER":
                    beta = min(beta, cached_score)
                if alpha >= beta:
                    return cached_score, cached_move

        # Generate and order moves
        candidates = move_selector.generate_candidates(board, limit=self.candidate_limit)
        if not candidates:
            return self._evaluate_terminal(board, node_color, current_score), None

        ordered_moves = self._order_moves(board, candidates, node_color, depth)

        # Main search loop
        alpha_orig = alpha
        best_score, best_local_move = self._search_moves(board, node_color, depth, alpha, beta, ordered_moves, current_score)
        
        # Store result in transposition table
        self._store_cache(key, depth, best_score, alpha_orig, beta, best_local_move)

        return best_score, best_local_move

    def _search_moves(self, board, node_color, depth, alpha, beta, ordered_moves, current_score):
        maximizing = (node_color == self.color)
        best_score = -INF if maximizing else INF
        best_local_move = None

        for move, win_now, _ in ordered_moves:
            x, y = move
            if node_color == -1 and renju_rules.is_forbidden(board, x, y, node_color):
                continue
            
            board.cells[y][x] = node_color
            board.move_count += 1

            try:
                new_score = heuristic.update_score_after_move(
                    board, x, y, node_color, self.color, current_score, patterns=self.patterns
                )
                if win_now:
                    score = (INF - board.move_count) if maximizing else (-INF + board.move_count)
                else:
                    score, _ = self._minimax(board, -node_color, depth - 1, alpha, beta, new_score, last_move=move)
            finally:
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
        
        return best_score, best_local_move

    def _evaluate_terminal(self, board, node_color, current_score):
        score = current_score
        if self.pv_helper:
            _, val = self.pv_helper.predict(board.cells, node_color)
            val_score = val if node_color == self.color else -val
            score += int(val_score * PV_SCALE)
        return score

    def _store_cache(self, key, depth, score, alpha_orig, beta, move):
        flag = "EXACT"
        if score <= alpha_orig:
            flag = "UPPER"
        elif score >= beta:
            flag = "LOWER"
        self.cache[key] = (depth, score, flag, move)

    def _order_moves(self, board, candidates, node_color, current_depth):
        pv_probs = None
        if self.pv_helper and current_depth == self.depth:
            probs, _ = self.pv_helper.predict(board.cells, node_color)
            pv_probs = probs

        ordered = []
        opp = -node_color
        for move in candidates:
            x, y = move
            if board.cells[y][x] != 0:
                continue
            
            board.cells[y][x] = node_color
            board.move_count += 1
            win_now = renju_rules.is_win_after_move(board, x, y, node_color)
            board.cells[y][x] = 0
            board.move_count -= 1
            
            local_score = INF // 2 if win_now else self._local_density(board, x, y, node_color, opp)

            if self.pv_move and move == self.pv_move and current_depth == self.depth:
                local_score += INF // 4
            if pv_probs is not None:
                idx = y * self.board_size + x
                local_score += pv_probs[idx].item() * (PV_SCALE / 2)
            
            ordered.append((move, win_now, local_score))

        ordered.sort(key=lambda item: (item[2], random.random()), reverse=True)
        return ordered

    def _local_density(self, board, x, y, node_color, opp):
        score = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not board.in_bounds(nx, ny):
                    continue
                
                val = board.cells[ny][nx]
                weight = 6 if abs(dx) <= 1 and abs(dy) <= 1 else 3
                if val == node_color:
                    score += weight
                elif val == opp:
                    score += weight - 2
                else: # empty
                    score += 1
        
        center = self.board_size // 2
        score -= abs(x - center) + abs(y - center) # favor center
        return score

    def _fallback_move(self, board):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board.is_empty(x, y):
                    if self.color == -1 and renju_rules.is_forbidden(board, x, y, self.color):
                        continue
                    return (x, y)
        return None # Should not happen on a non-full board

    def _record_stats(self):
        total_time = max(time.time() - self.start_time, 1e-9)
        self.stats_list.append({
            "color": self.color,
            "depth": self.depth,
            "nodes": self.node_counter,
            "time": total_time,
            "nps": self.node_counter / total_time,
        })

    def _solve_vcf(self, board, color, candidate_limit, max_depth=4):
        """Very narrow VCF-style search: look for double threats or immediate wins."""
        def time_guard():
            if time.time() > self.deadline:
                raise TimeoutError

        def immediate_wins(b, turn, cands):
            wins = []
            for mv in cands:
                x, y = mv
                if b.cells[y][x] != 0: continue
                if turn == -1 and renju_rules.is_forbidden(b, x, y, turn): continue
                
                b.cells[y][x] = turn
                b.move_count += 1
                if renju_rules.is_win_after_move(b, x, y, turn):
                    wins.append(mv)
                b.cells[y][x] = 0
                b.move_count -= 1
            return wins

        def dfs(b, turn, depth_left):
            time_guard()
            cands = move_selector.generate_candidates(b, limit=candidate_limit)
            win_moves = immediate_wins(b, turn, cands)
            if win_moves:
                return win_moves[0]
            if depth_left == 0:
                return None

            for mv in cands:
                x, y = mv
                if b.cells[y][x] != 0: continue
                if turn == -1 and renju_rules.is_forbidden(b, x, y, turn): continue
                
                b.cells[y][x] = turn
                b.move_count += 1
                
                # If this move creates a forced win sequence for the opponent
                if dfs(b, -turn, depth_left - 1) is None:
                    # Opponent has no response, so this is a winning move
                    b.cells[y][x] = 0
                    b.move_count -= 1
                    return mv
                
                b.cells[y][x] = 0
                b.move_count -= 1
            return None

        try:
            return dfs(board, color, max_depth)
        except TimeoutError:
            return None


def choose_move(board, color, depth, deadline, cache=None, zobrist_table=None, candidate_limit=15, patterns=None, pv_helper=None, stats=None, enable_vcf=False):
    """
    Public function to start a search. Instantiates and uses MinimaxSearcher.
    Maintains backward compatibility.
    """
    searcher = MinimaxSearcher(
        board_size=board.size,
        color=color,
        depth=depth,
        candidate_limit=candidate_limit,
        patterns=patterns,
        pv_helper=pv_helper,
        cache=cache,
        zobrist_table=zobrist_table,
        stats=stats,
        enable_vcf=enable_vcf
    )
    return searcher.choose_move(board, deadline)
