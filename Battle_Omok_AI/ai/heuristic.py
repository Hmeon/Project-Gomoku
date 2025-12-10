"""Pattern and weight definitions for Gomoku evaluation (open threes, fours, etc.)."""

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
    total = 0
    for line in _all_lines(board):
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
    for offset in range(4, 2 * size - 4):
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
