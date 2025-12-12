"""Symmetry augmentation tests for PV training samples."""

import torch

from Battle_Omok_AI.ai.dataset import SYMMETRY_TRANSFORMS, apply_symmetry_sample, encode_board


def test_symmetry_moves_pi_with_stone():
    size = 5
    x, y = 1, 3
    board = [[0] * size for _ in range(size)]
    board[y][x] = -1

    encoded = encode_board(board, to_play=-1)
    pi = torch.zeros(size * size, dtype=torch.float32)
    pi[y * size + x] = 1.0

    for symmetry_id in range(len(SYMMETRY_TRANSFORMS)):
        enc_t, pi_t = apply_symmetry_sample(encoded, pi, symmetry_id)

        stone_idx = int(enc_t[0].argmax().item())
        py, px = divmod(stone_idx, size)

        move_idx = int(pi_t.argmax().item())
        my, mx = divmod(move_idx, size)

        assert (px, py) == (mx, my)
        assert abs(float(pi_t.sum().item()) - 1.0) < 1e-6


def test_symmetry_rejects_bad_pi_length():
    size = 5
    board = [[0] * size for _ in range(size)]
    board[2][2] = -1
    encoded = encode_board(board, to_play=-1)
    pi = torch.zeros(size * size - 1, dtype=torch.float32)

    try:
        apply_symmetry_sample(encoded, pi, 0)
        assert False, "Expected ValueError for mismatched pi length"
    except ValueError as e:
        assert "pi length" in str(e)
