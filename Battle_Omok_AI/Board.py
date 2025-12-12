"""Board state container and victory checking (exactly five rule)."""


class Board:
    def __init__(self, size=15):
        # Store cells as -1 (black), 0 (empty), 1 (white)
        self.size = size
        self.cells = [[0] * size for _ in range(size)]
        self.move_count = 0
        self.history = []

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def is_empty(self, x, y):
        return self.in_bounds(x, y) and self.cells[y][x] == 0

    def place(self, x, y, color):
        """Place a stone; raise if out of bounds or occupied."""
        if color not in (-1, 1):
            raise ValueError("color must be -1 (black) or 1 (white)")
        if not self.in_bounds(x, y):
            raise ValueError("move out of bounds")
        if self.cells[y][x] != 0:
            raise ValueError("cell already occupied")
        self.cells[y][x] = color
        self.move_count += 1
        self.history.append((x, y))

    def clone(self):
        new_board = Board(self.size)
        new_board.cells = [row[:] for row in self.cells]
        new_board.move_count = self.move_count
        new_board.history = self.history[:]
        return new_board

    def has_exact_five(self, x, y):
        """Check for an exact five-in-a-row through (x, y); overlines do not win."""
        color = self.cells[y][x]
        if color not in (-1, 1):
            return False

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            forward = self._count_dir(x, y, dx, dy, color)
            backward = self._count_dir(x, y, -dx, -dy, color)
            total = 1 + forward + backward
            if total == 5:
                # Ensure no overline: ends must not extend the chain
                end1 = (x + dx * (forward + 1), y + dy * (forward + 1))
                end2 = (x - dx * (backward + 1), y - dy * (backward + 1))
                end1_same = self.in_bounds(*end1) and self.cells[end1[1]][end1[0]] == color
                end2_same = self.in_bounds(*end2) and self.cells[end2[1]][end2[0]] == color
                if not end1_same and not end2_same:
                    return True
        return False

    def has_five_or_more(self, x, y):
        """Check for 5+ in any direction through (x, y)."""
        color = self.cells[y][x]
        if color not in (-1, 1):
            return False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            forward = self._count_dir(x, y, dx, dy, color)
            backward = self._count_dir(x, y, -dx, -dy, color)
            total = 1 + forward + backward
            if total >= 5:
                return True
        return False

    def max_line_length(self, x, y):
        """Return the maximum contiguous line length through (x, y)."""
        color = self.cells[y][x]
        if color not in (-1, 1):
            return 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        best = 0
        for dx, dy in directions:
            forward = self._count_dir(x, y, dx, dy, color)
            backward = self._count_dir(x, y, -dx, -dy, color)
            best = max(best, 1 + forward + backward)
        return best

    def _count_dir(self, x, y, dx, dy, color):
        """Count contiguous stones of color from (x,y) (exclusive) in (dx,dy)."""
        count = 0
        cx, cy = x + dx, y + dy
        while self.in_bounds(cx, cy) and self.cells[cy][cx] == color:
            count += 1
            cx += dx
            cy += dy
        return count
