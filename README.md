diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..0858354d3ccfc4c2606039e9ba8274423c2098fa
--- /dev/null
+++ b/README.md
@@ -0,0 +1,65 @@
+# PersonalNanoGPT
+
+PersonalNanoGPT is a lightweight PyTorch implementation of a Transformer-based language model. It is intentionally small to allow experimentation on modest hardware while still demonstrating the components of a modern GPT-style architecture.
+
+## Features
+
+- Multi-head self-attention layers with causal masking for autoregressive generation.
+- Position-wise feed-forward networks and layer normalization for stable training.
+- Configurable hyperparameters such as context length, embedding size, and number of layers.
+- Simple training loop that tokenizes plain text and optimizes the model using AdamW.
+
+## Requirements
+
+- Python 3.8 or newer
+- PyTorch 2.0+
+
+Install PyTorch with pip (or use a pre-existing environment):
+
+```bash
+pip install torch
+```
+
+A CUDA-capable GPU is optional but accelerates training considerably.
+
+## Usage
+
+1. Prepare a UTF-8 text file for training. The dataset can be as small as a few kilobytes for testing.
+2. Run the training script with desired parameters:
+
+```bash
+python train.py --text data.txt --epochs 5 --batch-size 64 --block-size 128 --output gpt.pth
+```
+
+The command above trains for five epochs and saves the resulting weights to `gpt.pth`. The script automatically selects a GPU if one is available.
+
+### Customization
+
+- **block-size** controls the context window size. It must not exceed the model's `block_size` in `model.py`.
+- **batch-size** can be adjusted to fit GPU memory constraints.
+- **output** specifies a path to save model weights after training. If omitted, weights are not saved.
+
+## Generating Text
+
+After training, load the weights and sample from the model:
+
+```python
+import torch
+from model import GPT
+
+model = GPT(vocab_size=..., block_size=128)
+model.load_state_dict(torch.load('gpt.pth'))
+model.eval()
+```
+
+Use the loaded model to generate text by feeding it seed tokens and sampling the output distribution.
+
+## Repository Structure
+
+- `model.py` – Implementation of the GPT architecture.
+- `train.py` – Training loop and command-line interface.
+- `LICENSE` – Project license (MIT).
+
+## License
+
+This project is released under the MIT License. See `LICENSE` for the full text.
diff --git a/model.py b/model.py
new file mode 100644
index 0000000000000000000000000000000000000000..07d4975ec89d3dec4a8fb4ca82f2f7036a308b69
--- /dev/null
+++ b/model.py
@@ -0,0 +1,89 @@
+import torch
+import torch.nn as nn
+import math
+
+class SelfAttention(nn.Module):
+    def __init__(self, embed_dim: int, num_heads: int):
+        super().__init__()
+        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
+        self.num_heads = num_heads
+        self.head_dim = embed_dim // num_heads
+        self.query = nn.Linear(embed_dim, embed_dim)
+        self.key = nn.Linear(embed_dim, embed_dim)
+        self.value = nn.Linear(embed_dim, embed_dim)
+        self.proj = nn.Linear(embed_dim, embed_dim)
+
+    def forward(self, x: torch.Tensor):
+        B, T, C = x.size()
+        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
+        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
+        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
+
+        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
+        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
+        att = att.masked_fill(mask == 0, float('-inf'))
+        att = torch.softmax(att, dim=-1)
+
+        y = att @ v
+        y = y.transpose(1, 2).contiguous().view(B, T, C)
+        return self.proj(y)
+
+class FeedForward(nn.Module):
+    def __init__(self, embed_dim: int, hidden_dim: int):
+        super().__init__()
+        self.fc1 = nn.Linear(embed_dim, hidden_dim)
+        self.gelu = nn.GELU()
+        self.fc2 = nn.Linear(hidden_dim, embed_dim)
+
+    def forward(self, x: torch.Tensor):
+        return self.fc2(self.gelu(self.fc1(x)))
+
+class TransformerBlock(nn.Module):
+    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
+        super().__init__()
+        self.ln1 = nn.LayerNorm(embed_dim)
+        self.attn = SelfAttention(embed_dim, num_heads)
+        self.ln2 = nn.LayerNorm(embed_dim)
+        self.ff = FeedForward(embed_dim, hidden_dim)
+
+    def forward(self, x: torch.Tensor):
+        x = x + self.attn(self.ln1(x))
+        x = x + self.ff(self.ln2(x))
+        return x
+
+class GPT(nn.Module):
+    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8,
+                 num_layers: int = 4, hidden_dim: int = 512, block_size: int = 128):
+        super().__init__()
+        self.embed_dim = embed_dim
+        self.token_emb = nn.Embedding(vocab_size, embed_dim)
+        self.pos_emb = nn.Embedding(block_size, embed_dim)
+        self.blocks = nn.ModuleList([
+            TransformerBlock(embed_dim, num_heads, hidden_dim)
+            for _ in range(num_layers)
+        ])
+        self.ln_f = nn.LayerNorm(embed_dim)
+        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
+        self.block_size = block_size
+        self.apply(self._init_weights)
+
+    def _init_weights(self, module: nn.Module):
+        if isinstance(module, nn.Linear):
+            nn.init.normal_(module.weight, mean=0.0, std=0.02)
+            if module.bias is not None:
+                nn.init.zeros_(module.bias)
+        elif isinstance(module, nn.Embedding):
+            nn.init.normal_(module.weight, mean=0.0, std=0.02)
+
+    def forward(self, idx: torch.Tensor):
+        B, T = idx.size()
+        assert T <= self.block_size, "Sequence length exceeds block size"
+        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
+        tok_emb = self.token_emb(idx)
+        pos_emb = self.pos_emb(pos)
+        x = tok_emb + pos_emb
+        for block in self.blocks:
+            x = block(x)
+        x = self.ln_f(x)
+        logits = self.head(x)
+        return logits
diff --git a/train.py b/train.py
new file mode 100644
index 0000000000000000000000000000000000000000..f6f02b9a891a3dc6225ca18627bded99bf1b635a
--- /dev/null
+++ b/train.py
@@ -0,0 +1,69 @@
+import argparse
+from pathlib import Path
+
+import torch
+import torch.nn.functional as F
+from torch.utils.data import Dataset, DataLoader
+
+from model import GPT
+
+class TextDataset(Dataset):
+    def __init__(self, path: Path, block_size: int = 128):
+        data = path.read_text(encoding='utf-8')
+        chars = sorted(set(data))
+        self.stoi = {ch: i for i, ch in enumerate(chars)}
+        self.itos = {i: ch for ch, i in self.stoi.items()}
+        self.vocab_size = len(chars)
+        self.block_size = block_size
+        encoded = [self.stoi[ch] for ch in data]
+        self.data = torch.tensor(encoded, dtype=torch.long)
+
+    def __len__(self):
+        return len(self.data) - self.block_size
+
+    def __getitem__(self, idx):
+        chunk = self.data[idx: idx + self.block_size + 1]
+        return chunk[:-1], chunk[1:]
+
+
+def train(args):
+    text_path = Path(args.text)
+    ds = TextDataset(text_path, block_size=args.block_size)
+    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
+
+    device = 'cuda' if torch.cuda.is_available() else 'cpu'
+    model = GPT(ds.vocab_size, block_size=args.block_size).to(device)
+    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
+
+    for epoch in range(args.epochs):
+        for x, y in loader:
+            x, y = x.to(device), y.to(device)
+            logits = model(x)
+            loss = F.cross_entropy(logits.view(-1, ds.vocab_size), y.view(-1))
+            optim.zero_grad()
+            loss.backward()
+            optim.step()
+        print(f"Epoch {epoch+1}: loss {loss.item():.4f}")
+
+    if args.output:
+        torch.save(model.state_dict(), args.output)
+
+
+def main():
+    p = argparse.ArgumentParser(description='Train a minimal GPT model')
+    p.add_argument('--text', type=str, required=True,
+                   help='Path to training text file')
+    p.add_argument('--epochs', type=int, default=1,
+                   help='Number of training epochs')
+    p.add_argument('--batch-size', type=int, default=64,
+                   help='Training batch size')
+    p.add_argument('--block-size', type=int, default=128,
+                   help='Context size for the model')
+    p.add_argument('--output', type=str,
+                   help='Optional path to save trained model')
+    args = p.parse_args()
+    train(args)
+
+
+if __name__ == '__main__':
+    main()
