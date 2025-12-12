"""Pattern and weight definitions for Gomoku evaluation (open threes, fours, etc.)."""

from pathlib import Path
import yaml

# Default patterns; can be overridden by loading config/patterns.yaml if desired.
DEFAULT_PATTERNS = [
    ("011110", 50000),  # open four
    ("211110", 20000),  # closed four
    ("011112", 20000),  # closed four
    ("01110", 8000),    # open three
    ("010110", 8000),
    ("011010", 8000),
    ("011100", 4000),   # broken three
    ("001110", 4000),
    ("0101110", 4000),
    ("0110110", 4000),
    ("001100", 1000),   # open two
    ("01010", 1000),
    ("010010", 1000),
]


def load_patterns(path="config/patterns.yaml"):
    """Load pattern weights from YAML; fallback to defaults on error/missing."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        # Allow running from repo root (e.g., `python -m Battle_Omok_AI.main`).
        candidate = Path(__file__).resolve().parents[1] / path
        if candidate.exists():
            path = candidate

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return DEFAULT_PATTERNS

    loaded = []
    for item in data.get("patterns", []):
        pat = item.get("pattern")
        score = item.get("score", 0)
        if not pat:
            continue
        if isinstance(pat, list):
            for p in pat:
                loaded.append((p, score))
        else:
            loaded.append((pat, score))
    return loaded or DEFAULT_PATTERNS


def score_board(board, color, patterns=None):
    """
    Simple pattern-based evaluation. Positive favors `color`, negative favors opponent.
    board: Board instance
    color: -1 (black) or 1 (white)
    """
    patterns = patterns or DEFAULT_PATTERNS
    return score_lines(_all_lines(board), color, patterns)


def score_lines(lines, color, patterns=None):
    """Score a collection of lines for `color`."""
    patterns = patterns or DEFAULT_PATTERNS
    total = 0
    for line in lines:
        line_str_self, line_str_opp = _line_views(line, color)
        for pat, val in patterns:
            total += line_str_self.count(pat) * val
            total -= line_str_opp.count(pat) * val
    return total


def _all_lines(board):
    """Yield all rows, columns, and diagonals as lists of cell values."""
    size = board.size
    cells = board.cells

    # Rows and cols
    for y in range(size):
        yield cells[y]
    for x in range(size):
        yield [cells[y][x] for y in range(size)]

    # Diagonals (top-left to bottom-right)
    for offset in range(-size + 5, size - 4):  # minimum length 5
        diag = [cells[y][y - offset] for y in range(size) if 0 <= y - offset < size]
        if len(diag) >= 5:
            yield diag

    # Anti-diagonals (top-right to bottom-left)
    for offset in range(4, 2 * size - 5):
        anti = [cells[y][offset - y] for y in range(size) if 0 <= offset - y < size]
        if len(anti) >= 5:
            yield anti


def _line_views(line, color):
    """Return two strings: perspective of self and opponent."""
    opp = -color
    translate_self = {color: "1", opp: "2", 0: "0"}
    translate_opp = {color: "2", opp: "1", 0: "0"}
    to_self = "".join(translate_self[v] for v in line)
    to_opp = "".join(translate_opp[v] for v in line)
    return to_self, to_opp


def _line_with_override(board, x, y, dx, dy, override=None):
    """
    Collect a line through (x, y) along (dx, dy). If override is not None,
    the cell at (x, y) is replaced with that value in the returned line.
    """
    line = []
    cx, cy = x, y
    while board.in_bounds(cx - dx, cy - dy):
        cx -= dx
        cy -= dy
    while board.in_bounds(cx, cy):
        if cx == x and cy == y and override is not None:
            val = override
        else:
            val = board.cells[cy][cx]
        line.append(val)
        cx += dx
        cy += dy
    return line


def lines_through(board, x, y, override=None):
    """Return the four lines (row/col/diagonals) passing through (x, y)."""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    return [_line_with_override(board, x, y, dx, dy, override=override) for dx, dy in directions]


def update_score_after_move(board, x, y, move_color, eval_color, prev_score, patterns=None):
    """
    Incrementally update evaluation score after placing a stone at (x, y).
    board is assumed to already contain the stone (move_color) at (x, y).
    eval_color is the perspective for scoring (searcher's color).
    """
    patterns = patterns or DEFAULT_PATTERNS
    # Score contribution of affected lines before and after the move
    lines_after = lines_through(board, x, y, override=None)
    lines_before = lines_through(board, x, y, override=0)

    delta = score_lines(lines_after, eval_color, patterns) - score_lines(lines_before, eval_color, patterns)
    return prev_score + delta
