"""Dataset and helpers for loading self-play JSONL (board, pi, value)."""

from __future__ import annotations

import json
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def encode_board(board: List[List[int]], to_play: int) -> torch.Tensor:
    """
    Encode board into 3 channels: black stones, white stones, to_play plane.
    board: list of lists with -1 (black), 0, 1 (white)
    to_play: -1 or 1
    """
    h = len(board)
    w = len(board[0])
    blacks = torch.zeros((h, w), dtype=torch.float32)
    whites = torch.zeros((h, w), dtype=torch.float32)
    for y in range(h):
        for x in range(w):
            v = board[y][x]
            if v == -1:
                blacks[y, x] = 1.0
            elif v == 1:
                whites[y, x] = 1.0
    to_play_plane = torch.full((h, w), 1.0 if to_play == 1 else -1.0, dtype=torch.float32)
    return torch.stack([blacks, whites, to_play_plane], dim=0)


class SelfPlayDataset(Dataset):
    """Loads self-play JSONL where each line has keys: board, to_play, pi, value."""

    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        board = item["board"]
        to_play = item["to_play"]
        pi = torch.tensor(item["pi"], dtype=torch.float32)
        value = torch.tensor([item["value"]], dtype=torch.float32)
        encoded = encode_board(board, to_play)
        return encoded, pi, value
