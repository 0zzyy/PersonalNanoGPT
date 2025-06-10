"""Training script using PyTorch DDP."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset import TextDatasetConfig, load_dataset
from .model import GPT, GPTConfig


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    warmup_steps: int = 1000
    gradient_accumulation: int = 1
    weight_decay: float = 0.01
    data_path: str | None = None
    block_size: int = 1024
    save_every: int = 1000


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train GPT-2 on text data")
    p.add_argument("--epochs", type=int, default=1, help="number of epochs")
    p.add_argument("--batch-size", type=int, default=8, help="batch size per GPU")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--warmup-steps", type=int, default=1000, help="lr warmup steps")
    p.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation")
    p.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    p.add_argument("--data-path", type=str, default=None, help="path to text files")
    p.add_argument("--block-size", type=int, default=1024, help="context window")
    p.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    args = p.parse_args()
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        gradient_accumulation=args.grad_accum,
        weight_decay=args.weight_decay,
        data_path=args.data_path,
        block_size=args.block_size,
        save_every=args.save_every,
    )


def setup_ddp() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def main() -> None:
    cfg = parse_args()
    setup_ddp()

    model_cfg = GPTConfig(block_size=cfg.block_size)
    model = GPT(model_cfg).cuda()
    model = DDP(model, device_ids=[dist.get_rank() % torch.cuda.device_count()])

    ds = load_dataset(TextDatasetConfig(data_path=cfg.data_path, block_size=cfg.block_size))
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        for step, (x,) in enumerate(dl):
            x = x.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                logits, loss = model(x, x)
                loss = loss / cfg.gradient_accumulation
            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % cfg.save_every == 0 and dist.get_rank() == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(model.module.state_dict(), f"checkpoints/gpt2_step{global_step}.pt")

                if dist.get_rank() == 0 and global_step % 50 == 0:
                    print(
                        f"step {global_step} epoch {epoch} loss {loss.item() * cfg.gradient_accumulation:.4f}"
                    )

    if dist.get_rank() == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.module.state_dict(), "checkpoints/gpt2_final.pt")

    cleanup_ddp()


if __name__ == "__main__":
    main()
