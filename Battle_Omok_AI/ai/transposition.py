"""Zobrist hashing and transposition table helpers."""

import random


def zobrist_init(size=19, seed=None):
    rng = random.Random(seed)
    return [[rng.getrandbits(64) for _ in range(2)] for _ in range(size * size)]


def hash_board(board, table):
    """Compute Zobrist hash for a Board (-1 black, 1 white)."""
    h = 0
    size = board.size
    for y in range(size):
        for x in range(size):
            v = board.cells[y][x]
            if v == 0:
                continue
            color_idx = 0 if v == -1 else 1
            h ^= table[y * size + x][color_idx]
    return h


def lookup(ttable, key):
    return ttable.get(key)


def store(ttable, key, value):
    ttable[key] = value
