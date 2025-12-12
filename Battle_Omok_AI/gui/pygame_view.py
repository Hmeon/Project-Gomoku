"""Pygame-based board renderer and input helper."""

import os
import time

class PygameView:
    # --- Constants ---
    COLOR_BACKGROUND = (40, 30, 20)
    COLOR_WOOD = (209, 179, 135)
    COLOR_GRID = (60, 40, 20)
    COLOR_TEXT = (230, 230, 230)
    COLOR_RED = (200, 0, 0)

    PANEL_HEIGHT = 80

    def __init__(self, board_size, asset_dir="assets", window_size=800):
        import pygame

        self.asset_board_size = 540
        self.asset_margin = 23
        self.board_size = board_size
        self.asset_dir = asset_dir
        self.window_size = window_size
        self._pygame = pygame

        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Gomoku-Pro Battle")

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)

        # The board surface is now smaller to accommodate the info panel
        self.board_display_size = window_size - self.PANEL_HEIGHT
        self.board_surface = self._build_board_surface(self.board_display_size)
        
        self.tile_size = (self.board_surface.get_width() - 2 * self.margin_px) / (board_size - 1)
        
        piece_px = int(self.tile_size * 0.9)
        self.black_piece = self._load_image("black_piece.png", (piece_px, piece_px))
        self.white_piece = self._load_image("white_piece.png", (piece_px, piece_px))
        
        # Create semi-transparent versions for the hover marker
        self.black_piece_hover = self.black_piece.copy()
        self.white_piece_hover = self.white_piece.copy()
        self.black_piece_hover.set_alpha(128)
        self.white_piece_hover.set_alpha(128)

        # The board is offset by the panel height
        self.board_origin = (
            (window_size - self.board_surface.get_width()) // 2,
            self.PANEL_HEIGHT + (self.board_display_size - self.board_surface.get_height()) // 2,
        )

    def _grid_origin(self):
        ox, oy = self.board_origin
        return ox + self.margin_px, oy + self.margin_px

    def _build_board_surface(self, size_px):
        pygame = self._pygame
        board_path = os.path.join(self.asset_dir, "board.jpg")

        if self.board_size == 15 and os.path.exists(board_path):
            surf = self._load_image("board.jpg", (size_px, size_px))
            self.margin_px = surf.get_width() * (self.asset_margin / self.asset_board_size)
        else:
            surf = pygame.Surface((size_px, size_px)).convert()
            surf.fill(self.COLOR_WOOD)
            self.margin_px = size_px * (self.asset_margin / self.asset_board_size)
            grid_start = self.margin_px
            grid_end = size_px - self.margin_px
            tile = (grid_end - grid_start) / (self.board_size - 1)
            for i in range(self.board_size):
                offset = grid_start + i * tile
                pygame.draw.line(surf, self.COLOR_GRID, (grid_start, offset), (grid_end, offset), 1)
                pygame.draw.line(surf, self.COLOR_GRID, (offset, grid_start), (offset, grid_end), 1)
        return surf

    def _load_image(self, filename, scale_to=None):
        path = os.path.join(self.asset_dir, filename)
        surf = self._pygame.image.load(path).convert_alpha()
        if scale_to:
            surf = self._pygame.transform.smoothscale(surf, scale_to)
        return surf
    
    def _draw_text(self, text, font, color, center_pos):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=center_pos)
        self.screen.blit(text_surface, text_rect)

    def _draw_stones(self, board):
        gx, gy = self._grid_origin()
        for y, row in enumerate(board.cells):
            for x, stone_color in enumerate(row):
                if stone_color == 0:
                    continue
                
                cx = gx + x * self.tile_size
                cy = gy + y * self.tile_size
                piece_surf = self.black_piece if stone_color == -1 else self.white_piece
                px = cx - piece_surf.get_width() / 2
                py = cy - piece_surf.get_height() / 2
                self.screen.blit(piece_surf, (px, py))

    def _draw_last_move_marker(self, last_move):
        if not last_move:
            return
        
        gx, gy = self._grid_origin()
        lx, ly = last_move
        cx = gx + lx * self.tile_size
        cy = gy + ly * self.tile_size
        
        # A simple red dot in the center of the piece
        self._pygame.draw.circle(self.screen, self.COLOR_RED, (cx, cy), self.tile_size * 0.2)

    def _draw_info_panel(self, current_player_color, game_result):
        # Panel background
        panel_rect = self._pygame.Rect(0, 0, self.window_size, self.PANEL_HEIGHT)
        self._pygame.draw.rect(self.screen, self.COLOR_GRID, panel_rect)

        # Determine message based on game state
        if game_result is not None:
            if game_result == -1: msg = "Black Wins!"
            elif game_result == 1: msg = "White Wins!"
            elif game_result == 0: msg = "Draw"
            else: msg = "Game Over"
            self._draw_text(msg, self.font_large, self.COLOR_TEXT, (self.window_size / 2, self.PANEL_HEIGHT / 2))
        else:
            player = "Black" if current_player_color == -1 else "White"
            msg = f"{player} to move"
            self._draw_text(msg, self.font_medium, self.COLOR_TEXT, (self.window_size / 2, self.PANEL_HEIGHT / 2))

    def render(self, board, last_move=None, current_player_color=None, game_result=None):
        self.screen.fill(self.COLOR_BACKGROUND)
        self.screen.blit(self.board_surface, self.board_origin)
        
        self._draw_stones(board)
        self._draw_last_move_marker(last_move)
        self._draw_info_panel(current_player_color, game_result)

        self._pygame.display.flip()

    def _get_coords_from_mouse(self, pos):
        mx, my = pos
        gx, gy = self._grid_origin()
        
        board_min_x = gx - self.tile_size / 2
        board_max_x = gx + self.tile_size * (self.board_size - 1) + self.tile_size / 2
        board_min_y = gy - self.tile_size / 2
        board_max_y = gy + self.tile_size * (self.board_size - 1) + self.tile_size / 2

        if not (board_min_x <= mx <= board_max_x and board_min_y <= my <= board_max_y):
            return None

        grid_x = int(round((mx - gx) / self.tile_size))
        grid_y = int(round((my - gy) / self.tile_size))
        
        if 0 <= grid_x < self.board_size and 0 <= grid_y < self.board_size:
            return grid_x, grid_y
        return None

    def _draw_hover_marker(self, player_color):
        pos = self._pygame.mouse.get_pos()
        coords = self._get_coords_from_mouse(pos)
        if coords:
            gx, gy = self._grid_origin()
            cx = gx + coords[0] * self.tile_size
            cy = gy + coords[1] * self.tile_size
            
            piece_surf = self.black_piece_hover if player_color == -1 else self.white_piece_hover
            px = cx - piece_surf.get_width() / 2
            py = cy - piece_surf.get_height() / 2
            self.screen.blit(piece_surf, (px, py))

    def wait_for_move(self, board, deadline, player_color):
        pygame = self._pygame
        while True:
            if time.time() > deadline:
                raise TimeoutError("Move exceeded allotted time")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise TimeoutError("Window closed")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    coords = self._get_coords_from_mouse(event.pos)
                    if coords:
                        return coords
            
            # Re-render the board with the hover marker
            self.render(board, last_move=board.history[-1] if board.history else None, current_player_color=player_color)
            self._draw_hover_marker(player_color)
            pygame.display.flip()
            
            pygame.time.delay(10)

    def close(self):
        self._pygame.quit()
