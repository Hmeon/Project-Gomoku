"""Renju rule enforcement: forbidden moves for Black, win detection."""

from contextlib import contextmanager

try:
    from Board import Board
except ImportError:
    from Battle_Omok_AI.Board import Board


@contextmanager
def _simulate(board: Board, x: int, y: int, color: int):
    board._push_stone(x, y, color)
    try:
        yield
    finally:
        board._pop_stone(x, y)


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


def _run_indices_after_fill(line: list[int], pos: int, color_val: int = 1) -> list[int]:
    """
    Return indices of the contiguous run of color_val that would include pos
    if pos were filled with color_val.
    """
    left = pos - 1
    while left >= 0 and line[left] == color_val:
        left -= 1
    right = pos + 1
    while right < len(line) and line[right] == color_val:
        right += 1
    return list(range(left + 1, right))


def _is_open_four_after_fill(line: list[int], pos: int, move_idx: int, color_val: int = 1) -> tuple[bool, list[int]]:
    """Check if filling pos yields an open four including move_idx. Returns (ok, run_indices)."""
    run = _run_indices_after_fill(line, pos, color_val=color_val)
    if move_idx not in run or len(run) != 4:
        return False, run
    left = run[0] - 1
    right = run[-1] + 1
    if left < 0 or right >= len(line):
        return False, run
    if line[left] != 0 or line[right] != 0:
        return False, run
    return True, run


def _is_exact_five_after_fill(line: list[int], pos: int, move_idx: int, color_val: int = 1) -> tuple[bool, list[int]]:
    """Check if filling pos yields an exact five including move_idx. Returns (ok, run_indices)."""
    run = _run_indices_after_fill(line, pos, color_val=color_val)
    if move_idx not in run or len(run) != 5:
        return False, run
    left = run[0] - 1
    right = run[-1] + 1
    left_same = left >= 0 and line[left] == color_val
    right_same = right < len(line) and line[right] == color_val
    if left_same or right_same:
        return False, run
    return True, run


def _count_open_threes(board: Board, x: int, y: int, color: int) -> int:
    """
    Count distinct open threes created by the stone already placed at (x, y).
    We count unique open-four potentials (after one more black move) that include
    the new stone, collapsing multiple winning points from the same three.
    """
    if color != -1:
        raise ValueError("_count_open_threes is only defined for black (-1)")
    open_three_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        line, move_idx = _line_with_index(board, x, y, dx, dy)
        # Translate to perspective: black=1, white=2, empty=0
        chars = [1 if v == color else (2 if v == -color else 0) for v in line]
        for pos, v in enumerate(chars):
            if v != 0:
                continue
            ok, run = _is_open_four_after_fill(chars, pos, move_idx, color_val=1)
            if not ok:
                continue
            key = tuple(i for i in run if i != pos)
            open_three_keys.add((dx, dy, key))
    return len(open_three_keys)


def _count_fours(board: Board, x: int, y: int, color: int) -> int:
    """
    Count distinct fours created by the stone already placed at (x, y).
    A four is a line that can become an exact five in one move; an open four with
    two winning points counts as a single four.
    """
    if color != -1:
        raise ValueError("_count_fours is only defined for black (-1)")
    four_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        line, move_idx = _line_with_index(board, x, y, dx, dy)
        chars = [1 if v == color else (2 if v == -color else 0) for v in line]
        for pos, v in enumerate(chars):
            if v != 0:
                continue
            ok, run = _is_exact_five_after_fill(chars, pos, move_idx, color_val=1)
            if not ok:
                continue
            key = tuple(i for i in run if i != pos)
            four_keys.add((dx, dy, key))
    return len(four_keys)


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

        fours = _count_fours(board, x, y, color)
        if fours >= 2:
            return True

    return False
