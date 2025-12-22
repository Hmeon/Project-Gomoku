"""Candidate move generation (boundary-based, top-N filtering)."""

from collections import defaultdict


NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

RUN_ENDPOINT_MIN = 3
RUN_ENDPOINT_BONUS = 1000  # large enough to survive top-N truncation


def _apply_run_endpoint_bonus(board, scores: defaultdict[tuple[int, int], int], *, min_run: int = RUN_ENDPOINT_MIN) -> None:
    """
    Boost endpoints of contiguous runs (length >= min_run) so tactical blocks/extensions
    survive global top-N truncation when the board has dense clusters elsewhere.
    """

    size = board.size
    cells = board.cells

    def boost_line(coords: list[tuple[int, int]]) -> None:
        i = 0
        while i < len(coords):
            x, y = coords[i]
            v = cells[y][x]
            if v == 0:
                i += 1
                continue

            start = i
            i += 1
            while i < len(coords):
                x2, y2 = coords[i]
                if cells[y2][x2] != v:
                    break
                i += 1
            end = i - 1
            run_len = end - start + 1
            if run_len < min_run:
                continue

            bonus = RUN_ENDPOINT_BONUS * run_len
            if start - 1 >= 0:
                lx, ly = coords[start - 1]
                if cells[ly][lx] == 0:
                    scores[(lx, ly)] += bonus
            if end + 1 < len(coords):
                rx, ry = coords[end + 1]
                if cells[ry][rx] == 0:
                    scores[(rx, ry)] += bonus

    # Rows and cols
    for y in range(size):
        boost_line([(x, y) for x in range(size)])
    for x in range(size):
        boost_line([(x, y) for y in range(size)])

    # Diagonals (top-left to bottom-right)
    for x0 in range(size):
        coords = []
        x, y = x0, 0
        while x < size and y < size:
            coords.append((x, y))
            x += 1
            y += 1
        if len(coords) >= min_run + 1:
            boost_line(coords)
    for y0 in range(1, size):
        coords = []
        x, y = 0, y0
        while x < size and y < size:
            coords.append((x, y))
            x += 1
            y += 1
        if len(coords) >= min_run + 1:
            boost_line(coords)

    # Anti-diagonals (top-right to bottom-left)
    for x0 in range(size):
        coords = []
        x, y = x0, size - 1
        while x < size and y >= 0:
            coords.append((x, y))
            x += 1
            y -= 1
        if len(coords) >= min_run + 1:
            boost_line(coords)
    for y0 in range(size - 2, -1, -1):
        coords = []
        x, y = 0, y0
        while x < size and y >= 0:
            coords.append((x, y))
            x += 1
            y -= 1
        if len(coords) >= min_run + 1:
            boost_line(coords)


def _distance_ok(dx: int, dy: int, radius: int, metric: str) -> bool:
    """Return True if (dx, dy) falls within the chosen radius metric."""
    if metric == "manhattan":
        return abs(dx) + abs(dy) <= radius
    if metric == "euclidean":
        return dx * dx + dy * dy <= radius * radius
    if metric == "chebyshev":
        return max(abs(dx), abs(dy)) <= radius
    return True


def generate_candidates(
    board,
    last_move=None,
    limit=15,
    radius=2,
    *,
    distance_metric: str = "manhattan",
    color: int | None = None,
    patterns=None,
    use_heuristic: bool = False,
    pv_probs=None,
    pv_weight: float = 0.0,
):
    """
    Generate candidate empty cells near existing stones, ranked by a heuristic score.
    - If board empty: return center only.
    - Neighborhood: Manhattan/Euclidean radius (default manhattan radius=2).
    - Score = proximity + run-endpoint bonus + adjacency + optional pattern delta/prior weight.
    """
    occupied = getattr(board, "occupied", None)
    if occupied is None:
        occupied = [(x, y) for y in range(board.size) for x in range(board.size) if board.cells[y][x] != 0]

    if not occupied:
        center = board.size // 2
        return [(center, center)]

    size = board.size
    cells = board.cells
    scores: defaultdict[tuple[int, int], float] = defaultdict(float)

    for ox, oy in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if not _distance_ok(dx, dy, radius, distance_metric):
                    continue
                nx, ny = ox + dx, oy + dy
                if nx < 0 or nx >= size or ny < 0 or ny >= size:
                    continue
                if cells[ny][nx] != 0:
                    continue
                # Proximity: closer neighbors get higher incremental score
                if abs(dx) <= 1 and abs(dy) <= 1:
                    scores[(nx, ny)] += 2
                else:
                    scores[(nx, ny)] += 1

    _apply_run_endpoint_bonus(board, scores, min_run=RUN_ENDPOINT_MIN)

    # Extra boost for being adjacent to multiple stones
    for (nx, ny) in scores:
        adj_count = 0
        for dx, dy in NEIGHBORS_8:
            ax, ay = nx + dx, ny + dy
            if 0 <= ax < size and 0 <= ay < size and cells[ay][ax] != 0:
                adj_count += 1
        scores[(nx, ny)] += adj_count

    # Optional pattern-based delta scoring (for report-aligned move ordering)
    if use_heuristic and color in (-1, 1):
        try:
            from . import heuristic
        except ImportError:  # pragma: no cover - fallback for script mode
            from Battle_Omok_AI.ai import heuristic

        patterns = patterns or heuristic.DEFAULT_PATTERNS
        for nx, ny in list(scores):
            if cells[ny][nx] != 0:
                continue
            lines_after = heuristic.lines_through(board, nx, ny, override=color)
            lines_before = heuristic.lines_through(board, nx, ny, override=0)
            delta = heuristic.score_lines(lines_after, color, patterns) - heuristic.score_lines(lines_before, color, patterns)
            scores[(nx, ny)] += float(delta)

    # Optional policy prior weighting (root-level hint when available)
    if pv_probs is not None:
        weight = float(pv_weight)
        for nx, ny in list(scores):
            idx = ny * size + nx
            try:
                prior = pv_probs[idx].item()
            except AttributeError:
                prior = float(pv_probs[idx])
            scores[(nx, ny)] += prior * weight

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [pos for pos, _ in ranked[:limit]]
