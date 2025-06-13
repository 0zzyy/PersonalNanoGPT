# PersonalNanoGPT

PersonalNanoGPT is a compact implementation of a Transformer-based language model built with PyTorch. The project aims to provide a minimal yet functional codebase for experimentation and learning purposes.

## Features

- **Minimal GPT architecture** in `model.py` implemented with multi-head self-attention and feed-forward blocks.
- **Training script** `train.py` that tokenizes raw text and optimizes the model using AdamW.
- **Configurable parameters** including batch size, block size (context length), and number of training epochs.

## Requirements

- Python 3.8+
- PyTorch 2.0 or later

Install dependencies via pip:

```bash
pip install torch
```

## Usage

Prepare a plain text file to use as training data. Then run the training script:

```bash
python train.py --text path/to/data.txt --epochs 5 --batch-size 64 --block-size 128 --output gpt.pth
```

This command trains the model on `data.txt` for five epochs and saves the resulting weights to `gpt.pth`. If a GPU is available, the script will use it automatically.

## Repository Structure

- `model.py` – Defines the GPT model architecture.
- `train.py` – Provides a simple training loop for text data.
- `LICENSE` – MIT license information.

## License

This project is released under the MIT License. See `LICENSE` for details.
