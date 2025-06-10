"""Training script using PyTorch DDP."""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset import TextDatasetConfig, load_dataset
from .model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 on text data")
    p.add_argument("--epochs", type=int, default=1, help="number of epochs")
    p.add_argument("--batch-size", type=int, default=8, help="batch size per GPU")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--data-path", type=str, default=None, help="path to text files")
    p.add_argument("--block-size", type=int, default=1024, help="context window")
    return p.parse_args()


def setup_ddp() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    setup_ddp()

    config = GPTConfig(block_size=args.block_size)
    model = GPT(config).cuda()
    model = DDP(model, device_ids=[dist.get_rank() % torch.cuda.device_count()])

    ds = load_dataset(TextDatasetConfig(data_path=args.data_path, block_size=args.block_size))
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, (x,) in enumerate(dl):
            x = x.cuda(non_blocking=True)
            logits, loss = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 50 == 0 and dist.get_rank() == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if dist.get_rank() == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.module.state_dict(), "checkpoints/gpt2.pt")

    cleanup_ddp()


if __name__ == "__main__":
    main()
