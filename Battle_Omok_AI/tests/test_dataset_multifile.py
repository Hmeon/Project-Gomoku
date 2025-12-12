"""SelfPlayDataset multi-file and lazy-indexing tests."""

import json
import pickle

import torch

from Battle_Omok_AI.ai.dataset import SelfPlayDataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")


def test_dataset_loads_multiple_files_and_is_pickleable(tmp_path):
    size = 5
    empty = [[0] * size for _ in range(size)]
    pi = [0.0] * (size * size)
    pi[0] = 1.0

    file1 = tmp_path / "a.jsonl"
    file2 = tmp_path / "b.jsonl"

    rec1 = {"board": {"cells": empty}, "to_play": -1, "pi": pi, "value": 1}
    rec2 = {"board": {"cells": empty}, "to_play": 1, "pi": pi, "value": -1}
    rec3 = {"board": {"cells": empty}, "to_play": -1, "pi": pi, "value": 0}

    _write_jsonl(file1, [rec1, rec2])
    _write_jsonl(file2, [rec3, rec1, rec2])

    ds = SelfPlayDataset([file1, file2], augment=False)
    assert len(ds) == 5
    assert (tmp_path / "a.jsonl.idx").exists()
    assert (tmp_path / "b.jsonl.idx").exists()

    xb, pi_t, v = ds[0]
    assert xb.shape == (3, size, size)
    assert pi_t.shape == (size * size,)
    assert v.shape == (1,)

    # Ensure pickling works (important if DataLoader uses workers).
    ds2 = pickle.loads(pickle.dumps(ds))
    xb2, pi2, v2 = ds2[1]
    assert xb2.shape == (3, size, size)
    assert torch.allclose(pi2, pi_t) or pi2.shape == pi_t.shape
    assert v2.shape == (1,)


def test_dataset_skips_blank_and_bad_json_lines(tmp_path):
    size = 5
    empty = [[0] * size for _ in range(size)]
    pi = [0.0] * (size * size)
    pi[0] = 1.0

    path = tmp_path / "bad.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write("{")  # bad json
        f.write("\n")
        f.write(json.dumps({"board": {"cells": empty}, "to_play": -1, "pi": pi, "value": 1}))
        f.write("\n")

    ds = SelfPlayDataset(path, augment=False)
    assert len(ds) == 1
    assert (tmp_path / "bad.jsonl.idx").exists()


def test_dataset_index_cache_invalidation_on_file_change(tmp_path):
    size = 5
    empty = [[0] * size for _ in range(size)]
    pi = [0.0] * (size * size)
    pi[0] = 1.0

    path = tmp_path / "replay.jsonl"
    _write_jsonl(
        path,
        [
            {"board": {"cells": empty}, "to_play": -1, "pi": pi, "value": 1},
            {"board": {"cells": empty}, "to_play": 1, "pi": pi, "value": -1},
        ],
    )

    ds1 = SelfPlayDataset(path, augment=False)
    assert len(ds1) == 2

    idx_path = tmp_path / "replay.jsonl.idx"
    header1 = idx_path.read_text(encoding="utf-8").splitlines()[0]

    # Append one more valid record; cache should be invalidated and rewritten.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"board": {"cells": empty}, "to_play": -1, "pi": pi, "value": 0}))
        f.write("\n")

    ds2 = SelfPlayDataset(path, augment=False)
    assert len(ds2) == 3

    header2 = idx_path.read_text(encoding="utf-8").splitlines()[0]
    assert header1 != header2
