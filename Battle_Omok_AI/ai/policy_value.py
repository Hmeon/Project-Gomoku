"""Inference helper: load PV checkpoint and predict policy/value."""

from __future__ import annotations

import torch

from ai.dataset import encode_board
from ai.pv_model import PolicyValueNet


class PolicyValueInfer:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        args = ckpt.get("args", {})
        board_size = args.get("board_size", 15)
        channels = args.get("channels", 64)
        blocks = args.get("blocks", 5)
        self.model = PolicyValueNet(board_size=board_size, channels=channels, num_blocks=blocks).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.board_size = board_size

    @torch.no_grad()
    def predict(self, board, to_play: int):
        """
        board: 2D list of ints {-1,0,1}; to_play: -1 or 1
        Returns policy probs over board_size*board_size and scalar value.
        """
        x = encode_board(board, to_play).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        return probs, value.item()
