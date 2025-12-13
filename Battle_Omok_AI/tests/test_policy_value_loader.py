"""PolicyValueInfer robustness tests."""

import torch

from Battle_Omok_AI.ai.pv_model import PolicyValueNet
from Battle_Omok_AI.ai.policy_value import PolicyValueInfer


def test_load_plain_state_dict(tmp_path):
    # Save a plain state_dict without wrapper to ensure loader accepts it.
    # Must match PolicyValueInfer defaults (size=15, channels=64, blocks=5)
    # because plain state_dict doesn't carry config.
    model = PolicyValueNet(board_size=15, channels=64, num_blocks=5)
    state = model.state_dict()

    path = tmp_path / "plain_state.pt"
    path = str(path)

    torch.save(state, path)

    infer = PolicyValueInfer(path, device="cpu")
    assert infer.model.board_size == 15


def test_predict_raises_on_size_mismatch(tmp_path):
    model = PolicyValueNet(board_size=15, channels=64, num_blocks=5)
    state = model.state_dict()
    path = tmp_path / "plain_state.pt"
    torch.save(state, str(path))

    infer = PolicyValueInfer(str(path), device="cpu")
    small_board = [[0] * 9 for _ in range(9)]

    try:
        infer.predict(small_board, to_play=-1)
        assert False, "Expected ValueError for size mismatch"
    except ValueError as e:
        assert "does not match PV model" in str(e)


def test_predict_value_matches_predict_value_head(tmp_path):
    model = PolicyValueNet(board_size=15, channels=64, num_blocks=5)
    state = model.state_dict()
    path = tmp_path / "plain_state.pt"
    torch.save(state, str(path))

    infer = PolicyValueInfer(str(path), device="cpu")
    empty_board = [[0] * 15 for _ in range(15)]

    _, v1 = infer.predict(empty_board, to_play=-1)
    v2 = infer.predict_value(empty_board, to_play=-1)
    assert abs(v1 - v2) < 1e-6
