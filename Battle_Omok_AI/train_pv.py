"""Offline training loop for policy/value net from self-play JSONL."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ai.dataset import SelfPlayDataset
from ai.pv_model import PolicyValueNet


def loss_fn(logits, target_pi, values, target_v):
    # Support both one-hot and soft MCTS distributions.
    log_probs = F.log_softmax(logits, dim=1)
    target_pi = target_pi.clamp(min=0.0)
    target_pi = target_pi / target_pi.sum(dim=1, keepdim=True).clamp(min=1e-8)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_v)
    return policy_loss + value_loss, policy_loss, value_loss


def log_metrics(epoch, total_loss, policy_loss, value_loss):
    """Append training metrics to a CSV file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "train_metrics.csv"
    
    file_exists = log_file.exists()
    
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "epoch", "total_loss", "policy_loss", "value_loss"])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            f"{total_loss:.4f}",
            f"{policy_loss:.4f}",
            f"{value_loss:.4f}"
        ])


def train(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Device selection logic
    device = torch.device("cpu")
    is_dml = False
    if args.cpu:
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "dml":
        try:
            import torch_directml
            device = torch_directml.device()
            is_dml = True
            print("Using DirectML device")
        except ImportError:
            print("Warning: torch-directml not installed. Falling back to CPU.")
            device = torch.device("cpu")
    elif args.device and args.device != "cpu":
        device = torch.device(args.device)

    ds = SelfPlayDataset(
        args.data,
        augment=args.augment,
        augment_prob=args.augment_prob,
        seed=args.seed,
    )
    if len(ds) == 0:
        raise ValueError(f"No training samples found in {args.data}; run selfplay.py first.")

    use_batchnorm = not args.no_batchnorm and not is_dml

    # DirectML + BatchNorm can misbehave on very small remainder batches; drop_last avoids 1-sample tail.
    batch_size = min(args.batch_size, len(ds))
    drop_last = is_dml
    if args.workers < 0:
        raise ValueError(f"--workers must be >= 0, got {args.workers}")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=drop_last,
        pin_memory=bool(args.pin_memory),
    )

    model = PolicyValueNet(
        board_size=args.board_size,
        channels=args.channels,
        num_blocks=args.blocks,
        use_batchnorm=use_batchnorm,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total, pol_sum, val_sum = 0.0, 0.0, 0.0
        seen = 0
        model.train()
        for xb, pi, v in loader:
            xb = xb.to(device)
            pi = pi.to(device)
            v = v.to(device).squeeze(-1)
            batch_n = xb.size(0)
            opt.zero_grad()
            logits, values = model(xb)
            loss, pol_loss, val_loss = loss_fn(logits, pi, values, v)
            loss.backward()
            opt.step()
            total += loss.item() * batch_n
            pol_sum += pol_loss.item() * batch_n
            val_sum += val_loss.item() * batch_n
            seen += batch_n
        denom = max(seen, 1)
        avg_loss = total / denom
        avg_pol = pol_sum / denom
        avg_val = val_sum / denom
        print(f"epoch {epoch}: loss={avg_loss:.4f} policy={avg_pol:.4f} value={avg_val:.4f}")
        
        # Log metrics to CSV
        log_metrics(epoch, avg_loss, avg_pol, avg_val)

        torch.save(
            {"model_state": model.state_dict(), "args": vars(args) | {"use_batchnorm": use_batchnorm}},
            out_path,
        )
    print(f"Saved checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train policy/value net from self-play JSONL")
    parser.add_argument("--data", nargs="+", required=True, help="Path(s) to selfplay jsonl")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    parser.add_argument("--pin-memory", action="store_true", help="Pin memory for faster GPU transfers")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply random rotations/flips to (board, pi) on the fly (recommended).",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=1.0,
        help="Probability of applying symmetry augmentation per sample (default 1.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for shuffling/augmentation (optional)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--device", default="cpu", help="Device to use: cpu, cuda, or dml (DirectML for AMD)")
    parser.add_argument("--no-batchnorm", action="store_true", help="Disable BatchNorm layers (helpful for DML backends)")
    parser.add_argument("--output", default="checkpoints/pv_latest.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
