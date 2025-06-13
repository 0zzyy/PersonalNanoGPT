import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import GPT

class TextDataset(Dataset):
    def __init__(self, path: Path, block_size: int = 128):
        data = path.read_text(encoding='utf-8')
        chars = sorted(set(data))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        encoded = [self.stoi[ch] for ch in data]
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def train(args):
    text_path = Path(args.text)
    ds = TextDataset(text_path, block_size=args.block_size)
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


def main():
    p = argparse.ArgumentParser(description='Train a minimal GPT model')
    p.add_argument('--text', type=str, required=True,
                   help='Path to training text file')
    p.add_argument('--epochs', type=int, default=1,
                   help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Training batch size')
    p.add_argument('--block-size', type=int, default=128,
                   help='Context size for the model')
    p.add_argument('--output', type=str,
                   help='Optional path to save trained model')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
