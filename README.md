# PersonalNanoGPT

PersonalNanoGPT is a minimal, lightweight PyTorch implementation of a Transformer-based language model. It is intentionally small to allow experimentation on modest hardware while still demonstrating the components of a modern GPT-style architecture: multi-head self-attention with causal masking, position-wise feed-forward networks with GELU activations, layer normalization, and autoregressive generation.

## Features

- Multi-head self-attention layers with causal masking for autoregressive generation  
- Position-wise feed-forward networks and layer normalization for stable training  
- Configurable hyperparameters: context length (`block-size`), embedding size, number of layers, number of heads
- Simple training loop that tokenizes plain text and optimizes the model using AdamW
- Easy inference: load saved weights and generate text with autoregressive sampling
- Offline RL training with reward-weighted updates
- Reward-augmented decoding for domain control
- Self-alignment via domain term rewards
- Temperature-based calibration for confidence estimates

## Requirements

- Python 3.8 or newer  
- PyTorch 2.0+  

Install PyTorch via pip:

```bash
pip install torch
```

*A CUDA-capable GPU is optional but recommended for faster training.*

## Usage

1. Prepare a UTF-8 text file for training (e.g., `data.txt`).  
2. Train the model:

```bash
python train.py \
  --text data.txt \
  --epochs 5 \
  --batch-size 64 \
  --block-size 128 \
 --output gpt.pth
```

- `--epochs`: number of training epochs (default: 1)  
- `--batch-size`: training batch size (default: 64)  
- `--block-size`: context length (default: 128)  
- `--output`: path to save trained model weights (optional)

### Offline RL Training

Use `advanced.py` to train from a dataset containing text and rewards separated by `|`:

```bash
python advanced.py \
  --offline-data reward_data.txt \
  --epochs 3 \
  --output gpt_offline.pth
```

### Knowledge Discovery Training

`explore.py` provides a lightweight interface for personalised data exploration
on structured (`.csv`), semi-structured (`.jsonl`) and unstructured (`.txt`)
datasets.

```bash
python explore.py \
  --data path/to/data.csv \
  --epochs 2 \
  --output gpt_knowledge.pth
```

3. Generate text after training:

```python
import torch
from model import GPT

# Load model
model = GPT(vocab_size=..., block_size=128)
model.load_state_dict(torch.load('gpt.pth', map_location='cpu'))
model.eval()

# Prepare seed tokens (tensor of shape [1, context_length])
# e.g., idx = torch.tensor([[stoi[ch] for ch in "Once upon a time"]], dtype=torch.long)

# Generate new tokens
out = model.generate(idx, max_new_tokens=100, temperature=1.0)

# Convert token IDs back to text
generated = ''.join([itos[i.item()] for i in out[0]])
print(generated)
```

For reward-guided generation:

```python
reward_fn = char_reward_fn(['g'], stoi)
out = model.generate_reward_augmented(idx, max_new_tokens=50,
                                      reward_fn=reward_fn, beta=0.5)
```

## Repository Structure

```
.
├── model.py      # GPT architecture: TransformerBlock, SelfAttention, FeedForward, GPT class
├── train.py      # Training loop and CLI interface
├── LICENSE       # MIT License
└── README.md     # This file
```

## License

This project is released under the MIT License. See `LICENSE` for full text.
