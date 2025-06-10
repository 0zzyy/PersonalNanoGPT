# Personal NanoGPT

This repository contains a research‑oriented implementation for training a
GPT‑2 style language model. The goal is to provide a clear yet flexible code
base that can be extended for advanced experiments. The default configuration
matches the 124M parameter GPT‑2 model and supports distributed training on a
single node with eight A100 (40 GB) GPUs.

The repository contains the following modules:

* `src/dataset.py` – dataset loading utilities with optional streaming mode and
  GPT‑2 byte pair encoding.
* `src/model.py` – a concise implementation of the GPT‑2 architecture.
* `src/train.py` – a distributed training script using PyTorch DDP with mixed
  precision, gradient accumulation and cosine learning‑rate scheduling.
* `tests/` – unit tests verifying tokenization and model forward computations.

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare a dataset**

   The training script can download OpenWebText automatically or process a
   directory of plain text files. For very large corpora, enable streaming mode
   with the `--streaming` flag to avoid storing the entire dataset on disk.

3. **Launch training**

   Use `torchrun` to spawn one process per GPU. The following command starts an
   eight‑GPU run with mixed precision and gradient accumulation:

   ```bash
   torchrun --standalone --nproc_per_node=8 src/train.py --epochs 3 \
       --batch-size 8 --grad-accum 4 --warmup-steps 2000
   ```

   Training the 124M model on eight A100 GPUs typically finishes within four
   days when using the full OpenWebText dataset.

4. **Testing**

   Run the included unit tests to verify the installation:

   ```bash
   pytest -q
   ```

## Files

Refer to the extensive inline documentation for details about model
configuration, optimizer settings and dataset handling. The code is intentionally
modular so that new layers or training objectives can be added with minimal
changes.

