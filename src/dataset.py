"""Dataset utilities for training GPT models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import datasets
import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


@dataclass
class TextDatasetConfig:
    data_path: Optional[str] = None
    block_size: int = 1024
    num_files: Optional[int] = None


def load_dataset(config: TextDatasetConfig) -> Dataset:
    """Load and tokenize text data.

    If ``config.data_path`` is ``None`` the function downloads the
    ``openwebtext`` dataset. When ``data_path`` is specified, all files in that
    directory are read as plain text. ``num_files`` can be used to limit the
    number of files for quicker experiments.
    """
    tok = Tokenizer()

    if config.data_path is None:
        ds = datasets.load_dataset("openwebtext", split="train")
        text_iter: Iterable[str] = ds["text"]
    else:
        files = [os.path.join(config.data_path, f) for f in os.listdir(config.data_path)]
        if config.num_files:
            files = files[: config.num_files]
        text_iter = []
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                text_iter.append(fh.read())

    tokenized = []
    for text in text_iter:
        ids = tok.encode(text)
        for i in range(0, len(ids) - config.block_size, config.block_size):
            chunk = ids[i : i + config.block_size]
            tokenized.append(torch.tensor(chunk, dtype=torch.long))

    return torch.utils.data.TensorDataset(torch.stack(tokenized))
