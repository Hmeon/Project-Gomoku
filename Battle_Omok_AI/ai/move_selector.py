"""Candidate move generation (boundary-based, top-N filtering)."""

from collections import defaultdict


def generate_candidates(board, last_move=None, limit=15, radius=2):
    """
    Generate candidate empty cells near existing stones, ranked by proximity score.
    - If board empty: return center only.
    - Score = number of neighboring stones within 1 step (8-neighborhood).
    """
    occupied = [(x, y) for y in range(board.size) for x in range(board.size) if board.cells[y][x] != 0]
    if not occupied:
        center = board.size // 2
        return [(center, center)]

    scores = defaultdict(int)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for ox, oy in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = ox + dx, oy + dy
                if not board.in_bounds(nx, ny) or not board.is_empty(nx, ny):
                    continue
                # Proximity: closer neighbors get higher incremental score
                if abs(dx) <= 1 and abs(dy) <= 1:
                    scores[(nx, ny)] += 2
                else:
                    scores[(nx, ny)] += 1

    # Extra boost for being adjacent to multiple stones
    for (nx, ny) in list(scores.keys()):
        adj_count = 0
        for dx, dy in neighbors:
            ax, ay = nx + dx, ny + dy
            if board.in_bounds(ax, ay) and board.cells[ay][ax] != 0:
                adj_count += 1
        scores[(nx, ny)] += adj_count

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [pos for pos, _ in ranked[:limit]]
