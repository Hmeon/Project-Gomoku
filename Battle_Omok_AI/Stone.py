"""Data object for a single move (position and color)."""


class Stone:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
