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
PV_PRIOR_WEIGHT = PV_SCALE / 2
TIME_CHECK_MASK = 2047  # check every 2048 nodes
ORDER_RANK_WEIGHT = 3


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
        self._pv_value_cache = {}
        self._root_pv_probs = None

    def choose_move(self, board, deadline):
        """
        Return best move for color within depth and before deadline using iterative deepening.
        """
        self.deadline = deadline
        self.start_time = time.time()
        self.root_score = heuristic.score_board(board, self.color, patterns=self.patterns)
        root_hash = transposition.hash_board(board, self.zobrist_table)
        self._pv_value_cache = {}
        self._root_pv_probs = None
        if self.pv_helper is not None:
            try:
                probs, _ = self.pv_helper.predict(board.cells, self.color)
                self._root_pv_probs = probs
            except Exception:
                self._root_pv_probs = None

        # Tactical guardrails: immediate win or block before deeper search.
        win_move = self._find_immediate_win(board, self.color)
        if win_move is not None:
            return win_move
        block_move = self._find_immediate_block(board, self.color)
        if block_move is not None:
            return block_move
        
        # First try a narrow VCF search for forced wins, if enabled.
        if self.enable_vcf:
            vcf_move = self._solve_vcf(board, self.color, candidate_limit=min(self.candidate_limit, 12))
            if vcf_move is not None:
                return vcf_move

        best_move = None
        for current_depth in range(1, self.depth + 1):
            try:
                _, move = self._minimax(
                    board,
                    self.color,
                    current_depth,
                    -float("inf"),
                    float("inf"),
                    self.root_score,
                    last_move=None,
                    current_hash=root_hash,
                )
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

    def _find_immediate_win(self, board, color):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if not board.is_empty(x, y):
                    continue
                if color == -1 and renju_rules.is_forbidden(board, x, y, color):
                    continue
                board._push_stone(x, y, color)
                try:
                    if renju_rules.is_win_after_move(board, x, y, color):
                        return (x, y)
                finally:
                    board._pop_stone(x, y)
        return None

    def _find_immediate_block(self, board, color):
        opp = -color
        threat_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if not board.is_empty(x, y):
                    continue
                if opp == -1 and renju_rules.is_forbidden(board, x, y, opp):
                    continue
                board._push_stone(x, y, opp)
                try:
                    if renju_rules.is_win_after_move(board, x, y, opp):
                        threat_moves.append((x, y))
                finally:
                    board._pop_stone(x, y)

        if not threat_moves:
            return None

        for x, y in threat_moves:
            if color == -1 and renju_rules.is_forbidden(board, x, y, color):
                continue
            if board.is_empty(x, y):
                return (x, y)
        return None

    def _minimax(self, board, node_color, depth, alpha, beta, current_score, last_move, current_hash: int):
        self._time_ok()

        # Terminal state check
        if depth == 0 or board.move_count == self.board_size * self.board_size:
            return self._evaluate_terminal(board, node_color, current_score, current_hash), None

        # Transposition table lookup
        key = (current_hash, node_color)
        tt_move = None
        cached = self.cache.get(key)
        if cached:
            cached_depth, cached_score, cached_flag, cached_move = cached
            tt_move = cached_move
            if cached_depth >= depth:
                if cached_flag == "EXACT":
                    return cached_score, cached_move
                if cached_flag == "LOWER":
                    alpha = max(alpha, cached_score)
                elif cached_flag == "UPPER":
                    beta = min(beta, cached_score)
                if alpha >= beta:
                    return cached_score, cached_move

        # Generate and order moves (filter black fouls; fall back to full-scan if needed)
        root_pv_probs = self._root_pv_probs if (depth == self.depth and node_color == self.color) else None
        candidates = self._legal_candidates(board, node_color, pv_probs=root_pv_probs)
        if tt_move is not None:
            try:
                tx, ty = tt_move
                if board.is_empty(tx, ty) and not (node_color == -1 and renju_rules.is_forbidden(board, tx, ty, node_color)):
                    candidates = [tt_move] + [mv for mv in candidates if mv != tt_move]
                    candidates = candidates[: self.candidate_limit]
            except Exception:
                pass
        if not candidates:
            # No legal moves for the side to move -> disqualification loss.
            score = (-INF + board.move_count) if node_color == self.color else (INF - board.move_count)
            return score, None

        ordered_moves = self._order_moves(board, candidates, node_color, depth, tt_move=tt_move)
        if not ordered_moves:
            score = (-INF + board.move_count) if node_color == self.color else (INF - board.move_count)
            return score, None

        # Main search loop
        alpha_orig = alpha
        beta_orig = beta
        best_score, best_local_move = self._search_moves(
            board,
            node_color,
            depth,
            alpha,
            beta,
            ordered_moves,
            current_score,
            current_hash,
        )
        
        # Store result in transposition table
        self._store_cache(key, depth, best_score, alpha_orig, beta_orig, best_local_move)

        return best_score, best_local_move

    def _search_moves(self, board, node_color, depth, alpha, beta, ordered_moves, current_score, current_hash: int):
        maximizing = (node_color == self.color)
        best_score = -INF if maximizing else INF
        best_local_move = None

        for move, win_now, _ in ordered_moves:
            x, y = move
            
            board._push_stone(x, y, node_color)
            color_idx = 0 if node_color == -1 else 1
            next_hash = current_hash ^ self.zobrist_table[y * self.board_size + x][color_idx]

            try:
                if win_now:
                    score = (INF - board.move_count) if maximizing else (-INF + board.move_count)
                else:
                    new_score = heuristic.update_score_after_move(
                        board, x, y, node_color, self.color, current_score, patterns=self.patterns
                    )
                    score, _ = self._minimax(
                        board,
                        -node_color,
                        depth - 1,
                        alpha,
                        beta,
                        new_score,
                        last_move=move,
                        current_hash=next_hash,
                    )
            finally:
                board._pop_stone(x, y)

            if win_now:
                return score, move

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

    def _legal_candidates(self, board, node_color, pv_probs=None):
        """
        Generate candidate moves for a node.
        For black, filter forbidden moves; if all local candidates are forbidden,
        fall back to scanning all legal empties (rare but avoids search dead-ends).
        """
        candidates = move_selector.generate_candidates(
            board,
            limit=self.candidate_limit,
            color=node_color,
            patterns=self.patterns,
            use_heuristic=True,
            pv_probs=pv_probs,
            pv_weight=PV_PRIOR_WEIGHT,
        )
        if node_color != -1:
            return candidates

        legal = [mv for mv in candidates if not renju_rules.is_forbidden(board, mv[0], mv[1], node_color)]
        if legal:
            return legal

        # Rare fallback: expand to any legal empty cell.
        all_empty = [(x, y) for y in range(self.board_size) for x in range(self.board_size) if board.is_empty(x, y)]
        legal_all = [mv for mv in all_empty if not renju_rules.is_forbidden(board, mv[0], mv[1], node_color)]
        return legal_all[: self.candidate_limit]

    def _evaluate_terminal(self, board, node_color, current_score, current_hash: int):
        score = current_score
        if self.pv_helper:
            key = (current_hash, node_color)
            val = self._pv_value_cache.get(key)
            if val is None:
                if hasattr(self.pv_helper, "predict_value"):
                    val = self.pv_helper.predict_value(board.cells, node_color)
                else:
                    _, val = self.pv_helper.predict(board.cells, node_color)
                self._pv_value_cache[key] = val
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

    def _order_moves(self, board, candidates, node_color, current_depth, *, tt_move=None):
        pv_probs = None
        if self.pv_helper and current_depth == self.depth and self._root_pv_probs is None:
            try:
                probs, _ = self.pv_helper.predict(board.cells, node_color)
                pv_probs = probs
            except Exception:
                pv_probs = None

        ordered = []
        opp = -node_color
        rank_bonus = {
            mv: (len(candidates) - idx) * ORDER_RANK_WEIGHT
            for idx, mv in enumerate(candidates)
        }
        for move in candidates:
            x, y = move
            if board.cells[y][x] != 0:
                continue
            board._push_stone(x, y, node_color)
            try:
                win_now = renju_rules.is_win_after_move(board, x, y, node_color)
            finally:
                board._pop_stone(x, y)
            
            local_score = INF // 2 if win_now else self._local_density(board, x, y, node_color, opp)
            local_score += rank_bonus.get(move, 0)

            if tt_move is not None and move == tt_move:
                local_score += INF // 8
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
        raise ValueError("No legal moves available for search fallback")

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
        """
        Continuous-four (VCF) probe.

        This is a conservative forced-win shortcut used only at the root when
        enable_vcf=True. It searches for sequences where the attacker repeatedly
        creates a four threat (a next-move exact win), and the defender is forced
        to block those winning points. If the attacker can maintain this for up to
        max_depth plies, we treat the root move as a VCF win.
        """

        def time_guard():
            if time.time() > self.deadline:
                raise TimeoutError

        def immediate_wins(b, turn, cands):
            wins = []
            for mv in cands:
                x, y = mv
                if b.cells[y][x] != 0:
                    continue
                if turn == -1 and renju_rules.is_forbidden(b, x, y, turn):
                    continue

                b._push_stone(x, y, turn)
                try:
                    if renju_rules.is_win_after_move(b, x, y, turn):
                        wins.append(mv)
                finally:
                    b._pop_stone(x, y)
            return wins

        def legal_candidates(b, turn, limit):
            cands = move_selector.generate_candidates(b, limit=limit)
            if turn == -1:
                cands = [mv for mv in cands if not renju_rules.is_forbidden(b, mv[0], mv[1], turn)]
            return cands

        def winning_points(b, attacker):
            # Points where attacker would win immediately next move.
            cands = legal_candidates(b, attacker, limit=b.size * b.size)
            return immediate_wins(b, attacker, cands)

        def dfs(b, attacker, depth_left):
            time_guard()
            attack_moves = legal_candidates(b, attacker, limit=candidate_limit)
            if not attack_moves:
                return None

            for mv in attack_moves:
                time_guard()
                x, y = mv
                if b.cells[y][x] != 0:
                    continue

                b._push_stone(x, y, attacker)
                try:
                    if renju_rules.is_win_after_move(b, x, y, attacker):
                        return mv

                    points = winning_points(b, attacker)
                    if len(points) >= 2:
                        # Double-threat (open four / double four) -> forced win.
                        return mv
                    if depth_left == 0 or not points:
                        continue

                    defender = -attacker
                    defense_moves = []
                    for bx, by in points:
                        if b.cells[by][bx] != 0:
                            continue
                        if defender == -1 and renju_rules.is_forbidden(b, bx, by, defender):
                            continue
                        defense_moves.append((bx, by))

                    if len(defense_moves) < len(points):
                        # Some winning points cannot be legally blocked by defender.
                        return mv

                    forced = True
                    for block in defense_moves:
                        time_guard()
                        bx, by = block
                        b._push_stone(bx, by, defender)
                        try:
                            if renju_rules.is_win_after_move(b, bx, by, defender):
                                forced = False
                                break
                            if dfs(b, attacker, depth_left - 1) is None:
                                forced = False
                                break
                        finally:
                            b._pop_stone(bx, by)

                    if forced:
                        return mv
                finally:
                    b._pop_stone(x, y)

            return None

        try:
            return dfs(board, color, max_depth)
        except TimeoutError:
            return None


def choose_move(board, color, depth, deadline, cache=None, zobrist_table=None, candidate_limit=20, patterns=None, pv_helper=None, stats=None, enable_vcf=False):
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
