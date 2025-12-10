"""Abstract player interface for human or AI controllers."""


class Player:
    def __init__(self, color):
        self.color = color

    def next_move(self, board, deadline=None):
        """Return (x, y) for next move within time limit."""
        raise NotImplementedError


class HumanPlayer(Player):
    def __init__(self, color):
        super().__init__(color)

    def next_move(self, board, deadline=None):
        """Text-input player with deadline guard (raises TimeoutError on timeout)."""
        import sys
        import time
        import select

        prompt = "Enter move as 'x y' (0-indexed): "
        if deadline is None:
            raw = input(prompt).strip()
        else:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Move exceeded allotted time")
            rlist, _, _ = select.select([sys.stdin], [], [], remaining)
            if not rlist:
                raise TimeoutError("Move exceeded allotted time")
            raw = sys.stdin.readline().strip()

        try:
            x_str, y_str = raw.split()
            return int(x_str), int(y_str)
        except Exception:
            raise ValueError("Invalid input format; expected two integers")


class GuiHumanPlayer(Player):
    def __init__(self, color, view):
        super().__init__(color)
        self.view = view

    def next_move(self, board, deadline=None):
        if deadline is None:
            raise TimeoutError("GUI player requires deadline for responsiveness")
        return self.view.wait_for_move(deadline)
