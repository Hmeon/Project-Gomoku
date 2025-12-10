"""Pygame-based board renderer and input helper."""

import os
import time


class PygameView:
    def __init__(self, board_size, asset_dir="assets", window_size=800):
        import pygame

        # Board asset from gomokuAI-py-main: 540px square with ~23px padding to first grid line.
        self.asset_board_size = 540
        self.asset_margin = 23
        self.board_size = board_size
        self.asset_dir = asset_dir
        self.window_size = window_size
        self._pygame = pygame

        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Gomoku-Pro Battle")

        self.board_surface = self._build_board_surface(window_size)
        # Compute tile size from scaled board, honoring the margin baked into the image or the drawn grid.
        self.tile_size = (self.board_surface.get_width() - 2 * self.margin_px) / (board_size - 1)

        piece_px = int(self.tile_size * 0.9)
        self.black_piece = self._load_image("black_piece.png", (piece_px, piece_px))
        self.white_piece = self._load_image("white_piece.png", (piece_px, piece_px))

        # Center board if image is smaller than window
        self.board_origin = (
            (window_size - self.board_surface.get_width()) // 2,
            (window_size - self.board_surface.get_height()) // 2,
        )

    def _grid_origin(self):
        """Top-left grid intersection (image margin accounted for)."""
        ox, oy = self.board_origin
        return ox + self.margin_px, oy + self.margin_px

    def _build_board_surface(self, size_px):
        """Load board asset when available for 15x15; otherwise draw a grid so 19x19 works without an image."""
        pygame = self._pygame
        board_path = os.path.join(self.asset_dir, "board.jpg")

        if self.board_size == 15 and os.path.exists(board_path):
            surf = self._load_image("board.jpg", (size_px, size_px))
            self.margin_px = surf.get_width() * (self.asset_margin / self.asset_board_size)
            return surf

        # Generated board: light wood background with grid lines.
        surf = pygame.Surface((size_px, size_px)).convert()
        surf.fill((209, 179, 135))  # wood-like color
        self.margin_px = size_px * (self.asset_margin / self.asset_board_size)
        grid_start = self.margin_px
        grid_end = size_px - self.margin_px
        tile = (grid_end - grid_start) / (self.board_size - 1)
        grid_color = (60, 40, 20)
        for i in range(self.board_size):
            offset = grid_start + i * tile
            pygame.draw.line(surf, grid_color, (grid_start, offset), (grid_end, offset), 1)
            pygame.draw.line(surf, grid_color, (offset, grid_start), (offset, grid_end), 1)
        return surf

    def _load_image(self, filename, scale_to=None):
        path = os.path.join(self.asset_dir, filename)
        surf = self._pygame.image.load(path).convert_alpha()
        if scale_to:
            surf = self._pygame.transform.smoothscale(surf, scale_to)
        return surf

    def render(self, board, last_move=None):
        pygame = self._pygame
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.board_surface, self.board_origin)

        gx, gy = self._grid_origin()
        for y, row in enumerate(board.cells):
            for x, v in enumerate(row):
                if v == 0:
                    continue
                cx = gx + x * self.tile_size
                cy = gy + y * self.tile_size
                px = cx - self.black_piece.get_width() / 2
                py = cy - self.black_piece.get_height() / 2
                if v == -1:
                    self.screen.blit(self.black_piece, (px, py))
                elif v == 1:
                    self.screen.blit(self.white_piece, (px, py))

        if last_move:
            lx, ly = last_move
            cx = gx + lx * self.tile_size
            cy = gy + ly * self.tile_size
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (cx - self.tile_size / 2, cy - self.tile_size / 2, self.tile_size, self.tile_size),
                2,
            )

        pygame.display.flip()

    def wait_for_move(self, deadline):
        """Wait for a mouse click within deadline; return grid coords or raise TimeoutError."""
        pygame = self._pygame
        gx, gy = self._grid_origin()
        board_min_x = gx - self.tile_size / 2
        board_max_x = gx + self.tile_size * (self.board_size - 1) + self.tile_size / 2
        board_min_y = gy - self.tile_size / 2
        board_max_y = gy + self.tile_size * (self.board_size - 1) + self.tile_size / 2
        while True:
            if time.time() > deadline:
                raise TimeoutError("Move exceeded allotted time")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise TimeoutError("Window closed")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if not (board_min_x <= mx <= board_max_x and board_min_y <= my <= board_max_y):
                        continue
                    grid_x = int(round((mx - gx) / self.tile_size))
                    grid_y = int(round((my - gy) / self.tile_size))
                    if 0 <= grid_x < self.board_size and 0 <= grid_y < self.board_size:
                        return grid_x, grid_y
            pygame.time.delay(10)

    def close(self):
        self._pygame.quit()
