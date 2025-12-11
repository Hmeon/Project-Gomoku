"""Offline training loop for policy/value net from self-play JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ai.dataset import SelfPlayDataset
from ai.pv_model import PolicyValueNet


def loss_fn(logits, target_pi, values, target_v):
    policy_loss = F.cross_entropy(logits, target_pi.argmax(dim=1))
    value_loss = F.mse_loss(values, target_v)
    return policy_loss + value_loss, policy_loss, value_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = SelfPlayDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PolicyValueNet(board_size=args.board_size, channels=args.channels, num_blocks=args.blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total, pol_sum, val_sum = 0.0, 0.0, 0.0
        model.train()
        for xb, pi, v in loader:
            xb = xb.to(device)
            pi = pi.to(device)
            v = v.to(device).squeeze(-1)
            opt.zero_grad()
            logits, values = model(xb)
            loss, pol_loss, val_loss = loss_fn(logits, pi, values, v)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            pol_sum += pol_loss.item() * xb.size(0)
            val_sum += val_loss.item() * xb.size(0)
        n = len(ds)
        print(f"epoch {epoch}: loss={total/n:.4f} policy={pol_sum/n:.4f} value={val_sum/n:.4f}")
        torch.save(
            {"model_state": model.state_dict(), "args": vars(args)},
            out_path,
        )
    print(f"Saved checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train policy/value net from self-play JSONL")
    parser.add_argument("--data", required=True, help="Path to selfplay jsonl")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--output", default="checkpoints/pv_latest.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
