"""Inference helper: load PV checkpoint and predict policy/value."""

from __future__ import annotations

import torch

try:
    from ai.dataset import encode_board
    from ai.pv_model import PolicyValueNet
except ImportError:
    from .dataset import encode_board
    from .pv_model import PolicyValueNet


class PolicyValueInfer:
    def __init__(self, checkpoint: str, device: str | torch.device | None = None):
        if isinstance(device, torch.device):
            self.device = device
        elif device == "dml":
            try:
                import torch_directml
                self.device = torch_directml.device()
            except ImportError:
                print("Warning: torch-directml not found, using cpu")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        ckpt = torch.load(checkpoint, map_location=self.device)
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            args = ckpt.get("args", {})
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # Plain state_dict saved via torch.save(model.state_dict(), path)
            state_dict = ckpt
            args = {}
        else:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is missing 'model_state' and is not a valid state_dict. "
                "Re-save with torch.save({'model_state': model.state_dict(), 'args': {...}}, path)."
            )

        board_size = args.get("board_size", 15)
        channels = args.get("channels", 64)
        blocks = args.get("blocks", 5)
        use_batchnorm = args.get("use_batchnorm", True)
        # Backward compatibility: if bn keys are absent, disable batchnorm.
        if use_batchnorm and not any("bn" in k for k in state_dict.keys()):
            use_batchnorm = False
        self.model = PolicyValueNet(
            board_size=board_size,
            channels=channels,
            num_blocks=blocks,
            use_batchnorm=use_batchnorm,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.board_size = board_size

    @torch.no_grad()
    def predict(self, board, to_play: int):
        """
        board: 2D list of ints {-1,0,1}; to_play: -1 or 1
        Returns policy probs over board_size*board_size and scalar value.
        """
        cells = board["cells"] if isinstance(board, dict) and "cells" in board else board
        if not isinstance(cells, list) or not cells or not isinstance(cells[0], list):
            raise TypeError(f"Invalid board format for PV predict: {type(board)}")
        h, w = len(cells), len(cells[0])
        if h != self.board_size or w != self.board_size:
            raise ValueError(
                f"Board size {h}x{w} does not match PV model board_size {self.board_size}. "
                "Load a compatible checkpoint or disable PV."
            )

        x = encode_board(cells, to_play).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        return probs, value.item()
