"""Renju rule enforcement: forbidden moves for Black, win detection."""

from contextlib import contextmanager

try:
    from Board import Board
except ImportError:
    from Battle_Omok_AI.Board import Board


@contextmanager
def _simulate(board: Board, x: int, y: int, color: int):
    board.cells[y][x] = color
    board.move_count += 1
    try:
        yield
    finally:
        board.cells[y][x] = 0
        board.move_count -= 1


def is_win_after_move(board: Board, x: int, y: int, color: int) -> bool:
    """Assumes stone is already placed."""
    if color == -1:
        return board.has_exact_five(x, y)
    return board.has_five_or_more(x, y)


def _is_overline(board: Board, x: int, y: int, color: int) -> bool:
    return board.max_line_length(x, y) > 5


def _line_with_index(board: Board, x: int, y: int, dx: int, dy: int):
    """Return line values along direction and the index of (x, y) within that line."""
    line = []
    cx, cy = x, y
    # move to start of line
    while board.in_bounds(cx - dx, cy - dy):
        cx -= dx
        cy -= dy
    idx = 0
    while board.in_bounds(cx, cy):
        line.append(board.cells[cy][cx])
        if cx == x and cy == y:
            idx = len(line) - 1
        cx += dx
        cy += dy
    return line, idx


OPEN_FOUR = "011110"


def _count_open_threes(board: Board, x: int, y: int, color: int) -> int:
    """Count open threes created by the stone already placed at (x, y)."""
    assert color == -1
    total = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        line, idx = _line_with_index(board, x, y, dx, dy)
        chars = "".join("1" if v == color else ("2" if v == -color else "0") for v in line)
        for pos, ch in enumerate(chars):
            if ch != "0":
                continue
            # simulate filling this empty to test if it forms an open four in this line
            filled = chars[:pos] + "1" + chars[pos + 1 :]
            for start in range(max(0, pos - 5), min(len(filled) - 5, pos) + 1):
                segment = filled[start : start + 6]
                if segment == OPEN_FOUR and start <= pos <= start + 5:
                    total += 1
                    break
    return total


def _count_four_threats(board: Board, x: int, y: int, color: int) -> int:
    """Count positions (in any direction) where one more stone would make a five."""
    assert color == -1
    threats = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        line, _ = _line_with_index(board, x, y, dx, dy)
        chars = [1 if v == color else (2 if v == -color else 0) for v in line]
        for pos, v in enumerate(chars):
            if v != 0:
                continue
            # Fill and measure contiguous run including pos
            chars[pos] = 1
            run = 1
            # left
            i = pos - 1
            while i >= 0 and chars[i] == 1:
                run += 1
                i -= 1
            # right
            i = pos + 1
            while i < len(chars) and chars[i] == 1:
                run += 1
                i += 1
            if run >= 5:
                threats += 1
            chars[pos] = 0
    return threats


def is_forbidden(board: Board, x: int, y: int, color: int) -> bool:
    """Return True if placing here is a foul for Black. White is never forbidden."""
    if color != -1:
        return False

    if not board.is_empty(x, y):
        return True

    with _simulate(board, x, y, color):
        # Five has priority over fouls
        if board.has_exact_five(x, y):
            return False

        overline = _is_overline(board, x, y, color)
        if overline:
            return True

        open_threes = _count_open_threes(board, x, y, color)
        if open_threes >= 2:
            return True

        four_threats = _count_four_threats(board, x, y, color)
        if four_threats >= 2:
            return True

    return False
