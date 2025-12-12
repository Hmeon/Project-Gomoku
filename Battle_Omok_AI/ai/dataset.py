"""Dataset and helpers for loading self-play JSONL (board, pi, value)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


SYMMETRY_TRANSFORMS: tuple[tuple[int, bool], ...] = (
    (0, False),  # identity
    (1, False),  # rot90
    (2, False),  # rot180
    (3, False),  # rot270
    (0, True),   # flip-x
    (1, True),   # rot90 + flip-x
    (2, True),   # rot180 + flip-x
    (3, True),   # rot270 + flip-x
)


def encode_board(board: List[List[int]] | dict, to_play: int) -> torch.Tensor:
    """
    Encode board into 3 channels: black stones, white stones, to_play plane.
    """
    if isinstance(board, dict):
        if "cells" not in board:
            raise KeyError("Board snapshot dict must contain key 'cells'")
        board = board["cells"]

    if not isinstance(board, list) or not board:
        raise TypeError(f"Board must be a non-empty list[list[int]], got {type(board)}")
    if not isinstance(board[0], list) or not board[0]:
        raise TypeError("Board must be a non-empty list[list[int]]")
    if to_play not in (-1, 1):
        raise ValueError(f"to_play must be -1 or 1, got {to_play}")

    cells = torch.tensor(board, dtype=torch.int8)
    if cells.ndim != 2:
        raise ValueError(f"Board tensor must be 2D, got shape {tuple(cells.shape)}")

    blacks = (cells == -1).to(torch.float32)
    whites = (cells == 1).to(torch.float32)
    to_play_plane = torch.full(blacks.shape, 1.0 if to_play == 1 else -1.0, dtype=torch.float32)
    return torch.stack((blacks, whites, to_play_plane), dim=0)


def apply_symmetry_planes(x: torch.Tensor, symmetry_id: int) -> torch.Tensor:
    """
    Apply a dihedral symmetry to a tensor whose last 2 dims are (H, W).
    symmetry_id in [0, 7] as defined by SYMMETRY_TRANSFORMS.
    """
    if symmetry_id < 0 or symmetry_id >= len(SYMMETRY_TRANSFORMS):
        raise ValueError(f"symmetry_id must be in [0, {len(SYMMETRY_TRANSFORMS) - 1}], got {symmetry_id}")
    k, flip = SYMMETRY_TRANSFORMS[symmetry_id]
    out = torch.rot90(x, k, dims=(-2, -1)) if k else x
    if flip:
        out = torch.flip(out, dims=(-1,))
    return out


def apply_symmetry_sample(encoded: torch.Tensor, pi: torch.Tensor, symmetry_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same symmetry to encoded board [C,H,W] and policy [H*W]."""
    if encoded.ndim != 3:
        raise ValueError(f"encoded must be 3D [C,H,W], got shape {tuple(encoded.shape)}")
    h, w = encoded.shape[-2], encoded.shape[-1]
    if pi.numel() != h * w:
        raise ValueError(f"pi length {pi.numel()} does not match board {h}x{w}")

    pi2d = pi.view(h, w)
    encoded_t = apply_symmetry_planes(encoded, symmetry_id)
    pi_t = apply_symmetry_planes(pi2d, symmetry_id).reshape(-1)
    return encoded_t, pi_t


class SelfPlayDataset(Dataset):
    """Loads self-play JSONL where each line has keys: board, to_play, pi, value."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        augment: bool = False,
        augment_prob: float = 1.0,
        seed: int | None = None,
        cache_index: bool = True,
    ):
        self.paths = self._normalize_paths(paths)
        self.augment = augment
        self.augment_prob = float(augment_prob)
        if not 0.0 <= self.augment_prob <= 1.0:
            raise ValueError(f"augment_prob must be in [0,1], got {self.augment_prob}")
        self.rng = random.Random(seed)
        self.cache_index = bool(cache_index)

        self._index: list[tuple[int, int]] = []  # (file_idx, offset)
        self._handles: dict[int, object] = {}
        self._build_index()

    @staticmethod
    def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(paths, (str, Path)):
            return [Path(paths)]
        return [Path(p) for p in paths]

    def _build_index(self) -> None:
        """
        Build a random-access index of JSONL line offsets without storing samples in memory.
        This keeps memory usage stable even for large self-play datasets.
        """
        for file_idx, path in enumerate(self.paths):
            if self.cache_index and self._try_load_cached_index(file_idx, path):
                continue

            offsets: list[int] = []
            with open(path, "r", encoding="utf-8") as f:
                line_no = 0
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line_no += 1
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        LOGGER.warning("Skipping bad JSON in %s at line %d: %s", path, line_no, e)
                        continue
                    offsets.append(offset)

            self._index.extend((file_idx, off) for off in offsets)
            if self.cache_index:
                self._try_write_cached_index(path, offsets)

    @staticmethod
    def _index_path(path: Path) -> Path:
        suffix = f"{path.suffix}.idx" if path.suffix else ".idx"
        return path.with_suffix(suffix)

    def _try_load_cached_index(self, file_idx: int, path: Path) -> bool:
        idx_path = self._index_path(path)
        if not idx_path.exists():
            return False
        try:
            stat = path.stat()
            with open(idx_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                if not header.startswith("#"):
                    return False
                parts = header[1:].strip().split()
                meta = {}
                for part in parts:
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    meta[k] = v
                if int(meta.get("size", "-1")) != stat.st_size:
                    return False
                if int(meta.get("mtime_ns", "-1")) != getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)):
                    return False

                offsets: list[int] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    offsets.append(int(line))
        except Exception:
            return False

        self._index.extend((file_idx, off) for off in offsets)
        return True

    def _try_write_cached_index(self, path: Path, offsets: list[int]) -> None:
        idx_path = self._index_path(path)
        try:
            stat = path.stat()
            mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
            with open(idx_path, "w", encoding="utf-8") as f:
                f.write(f"# size={stat.st_size} mtime_ns={mtime_ns}\n")
                for off in offsets:
                    f.write(f"{off}\n")
        except OSError:
            return

    def _get_handle(self, file_idx: int):
        handle = self._handles.get(file_idx)
        if handle is None or getattr(handle, "closed", False):
            handle = open(self.paths[file_idx], "r", encoding="utf-8")
            self._handles[file_idx] = handle
        return handle

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if not handles:
            return
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, offset = self._index[idx]
        handle = self._get_handle(file_idx)
        handle.seek(offset)
        line = handle.readline()
        item = json.loads(line)
        board = item["board"]
        to_play = item["to_play"]
        pi = torch.tensor(item["pi"], dtype=torch.float32)
        value = torch.tensor([item["value"]], dtype=torch.float32)
        encoded = encode_board(board, to_play)
        if self.augment and self.rng.random() < self.augment_prob:
            symmetry_id = self.rng.randrange(len(SYMMETRY_TRANSFORMS))
            encoded, pi = apply_symmetry_sample(encoded, pi, symmetry_id)
        return encoded, pi, value
