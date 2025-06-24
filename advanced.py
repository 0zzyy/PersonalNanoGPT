import argparse
from pathlib import Path
from typing import List, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import GPT

class RewardTextDataset(Dataset):
    """Dataset of text samples paired with scalar rewards."""
    def __init__(self, path: Path, block_size: int = 128):
        lines = path.read_text(encoding='utf-8').splitlines()
        texts, rewards = [], []
        for line in lines:
            if '\t' in line:
                text, reward = line.rsplit('\t', 1)
            elif '|' in line:
                text, reward = line.rsplit('|', 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                text = ' '.join(parts[:-1])
                reward = parts[-1]
            texts.append(text)
            rewards.append(float(reward))
        chars = sorted(set(''.join(texts)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.samples = []
        for text, r in zip(texts, rewards):
            encoded = [self.stoi[ch] for ch in text]
            if len(encoded) < 2:
                continue
            x = torch.tensor(encoded[:-1], dtype=torch.long)
            y = torch.tensor(encoded[1:], dtype=torch.long)
            self.samples.append((x, y, r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, r = self.samples[idx]
        if x.size(0) > self.block_size:
            x = x[:self.block_size]
            y = y[:self.block_size]
        return x, y, torch.tensor(r, dtype=torch.float)


def collate_reward(batch: Iterable):
    xs, ys, rs = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.zeros(len(ys), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    r = torch.tensor(rs, dtype=torch.float)
    return x_pad, y_pad, r


def train_offline_rl(args: argparse.Namespace):
    data_path = Path(args.offline_data)
    ds = RewardTextDataset(data_path, block_size=args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_reward)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(ds.vocab_size, block_size=args.block_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        for x, y, r in loader:
            x, y, r = x.to(device), y.to(device), r.to(device)
            logits = model(x)
            loss_per_token = F.cross_entropy(
                logits.view(-1, ds.vocab_size), y.view(-1), reduction='none')
            loss_seq = loss_per_token.view(x.size(0), -1).mean(dim=1)
            loss = (loss_seq * r).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Offline RL Epoch {epoch+1}: loss {loss.item():.4f}")

    if args.output:
        torch.save(model.state_dict(), args.output)


def char_reward_fn(chars: List[str], stoi: dict) -> callable:
    """Return a reward function encouraging given characters."""
    ids = [stoi[ch] for ch in chars if ch in stoi]
    reward = torch.zeros(len(stoi))
    reward[ids] = 1.0
    def fn(idx: torch.Tensor):
        return reward.to(idx.device).unsqueeze(0).expand(idx.size(0), -1)
    return fn



def main():
    p = argparse.ArgumentParser(description="Offline RL training")
    p.add_argument('--offline-data', type=str, required=True,
                   help='Path to reward data file')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--output', type=str,
                   help='Optional path to save trained model')
    args = p.parse_args()
    train_offline_rl(args)


if __name__ == '__main__':
    main()
