# Personal NanoGPT

This project provides a minimal yet complete implementation for training a GPT-2
(124M parameter) language model on the OpenWebText dataset using PyTorch. The
code is designed for multi-GPU training on a single node with eight A100 GPUs.

The repository contains the following modules:

* `src/dataset.py` – utilities to download and tokenize text datasets using
the GPT‑2 BPE tokenizer.
* `src/model.py` – a lightweight implementation of the GPT‑2 architecture.
* `src/train.py` – the training loop using PyTorch Distributed Data Parallel
  (DDP) for efficient multi-GPU training.
* `tests/` – a small test suite ensuring that tokenization and model forward
  pass work correctly.

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare a dataset**

   By default the code downloads the `openwebtext` dataset from the
   Hugging Face Hub. If you have your own text dataset, place the files under a
   directory and pass the path with `--data-path` when running the training
   script.

3. **Launch training**

   The training script uses `torchrun` to spawn one process per GPU. The example
   below launches an eight‑GPU run:

   ```bash
   torchrun --standalone --nproc_per_node=8 src/train.py --epochs 3 --batch-size 8
   ```

   Training a 124M parameter model on eight A100 (40 GB) GPUs should finish
   within four days on the full OpenWebText dataset.

4. **Testing**

   Run unit tests with:

   ```bash
   pytest
   ```

## Files

Refer to the code comments in each file for more details about the
implementation and how to adjust settings such as the learning rate, sequence
length, or tokenizer. The goal is to keep the code easy to understand while
providing sensible defaults for high‑quality text generation.

