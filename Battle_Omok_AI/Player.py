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
        import os
        import sys
        import time

        prompt = "Enter move as 'x y' (0-indexed): "
        if deadline is None:
            raw = input(prompt).strip()
        else:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Move exceeded allotted time")

            if os.name == "nt":
                # Windows: select() on stdin is not supported. Poll with msvcrt.
                import msvcrt

                buffer = ""
                while time.time() < deadline:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwche()
                        if ch in ("\r", "\n"):
                            sys.stdout.write("\n")
                            break
                        buffer += ch
                    time.sleep(0.01)
                else:
                    raise TimeoutError("Move exceeded allotted time")
                raw = buffer.strip()
            else:
                import select

                rlist, _, _ = select.select([sys.stdin], [], [], remaining)
                if not rlist:
                    raise TimeoutError("Move exceeded allotted time")
                raw = sys.stdin.readline().strip()

        try:
            x_str, y_str = raw.split()
            return int(x_str), int(y_str)
        except ValueError as exc:
            raise ValueError("Invalid input format; expected two integers") from exc


class GuiHumanPlayer(Player):
    def __init__(self, color, view):
        super().__init__(color)
        self.view = view

    def next_move(self, board, deadline=None):
        if deadline is None:
            raise TimeoutError("GUI player requires deadline for responsiveness")
        return self.view.wait_for_move(board, deadline, self.color)
