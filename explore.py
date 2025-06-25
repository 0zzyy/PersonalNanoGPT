import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model import GPT

class KnowledgeDataset(Dataset):
    """Load text from txt, csv, or json/jsonl files for GPT training."""
    def __init__(self, path: Path, block_size: int = 128):
        data = self._read_file(path)
        chars = sorted(set(data))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        encoded = [self.stoi[ch] for ch in data]
        self.data = torch.tensor(encoded, dtype=torch.long)

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == '.txt':
            return path.read_text(encoding='utf-8')
        if path.suffix.lower() == '.csv':
            rows = []
            with path.open(newline='', encoding='utf-8') as f:
                for row in csv.reader(f):
                    rows.append(' '.join(row))
            return '\n'.join(rows)
        if path.suffix.lower() in {'.json', '.jsonl'}:
            lines = []
            with path.open(encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    lines.append(' '.join(map(str, self._flatten(obj))))
            return '\n'.join(lines)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _flatten(self, obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from self._flatten(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from self._flatten(item)
        else:
            yield obj

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def train_knowledge(args: argparse.Namespace):
    ds = KnowledgeDataset(Path(args.data), block_size=args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(ds.vocab_size, block_size=args.block_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, ds.vocab_size), y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

    if args.output:
        torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Knowledge Discovery Training')
    p.add_argument('--data', type=str, required=True,
                   help='Path to txt/csv/json data file')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--output', type=str,
                   help='Optional path to save trained model')
    args = p.parse_args()
    train_knowledge(args)
