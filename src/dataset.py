"""Dataset utilities for training GPT models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import datasets
import torch
from torch.utils.data import Dataset, IterableDataset

from .tokenizer import Tokenizer


@dataclass
class TextDatasetConfig:
    """Configuration for text dataset loading."""

    data_path: Optional[str] = None
    block_size: int = 1024
    num_files: Optional[int] = None
    streaming: bool = False
    dataset_name: str = "openwebtext"
    split: str = "train"


def load_dataset(config: TextDatasetConfig) -> Dataset:
    """Load and tokenize text data.

    Parameters
    ----------
    config:
        Dataset configuration controlling source, block size and streaming mode.

    Returns
    -------
    torch.utils.data.Dataset or torch.utils.data.IterableDataset
        Dataset of token IDs ready for training.
    """
    tok = Tokenizer()

    if config.data_path is None:
        ds = datasets.load_dataset(
            config.dataset_name,
            split=config.split,
            streaming=config.streaming,
        )
        text_iter: Iterable[str]
        if config.streaming:
            text_iter = (x["text"] for x in ds)  # type: ignore[index]
        else:
            text_iter = ds["text"]  # type: ignore[index]
    else:
        files = [os.path.join(config.data_path, f) for f in os.listdir(config.data_path)]
        if config.num_files:
            files = files[: config.num_files]
        text_iter = []
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                text_iter.append(fh.read())

    if config.streaming:
        class TokenIterableDataset(IterableDataset):
            def __iter__(self_inner):
                for text in text_iter:
                    ids = tok.encode(text)
                    for i in range(0, len(ids) - config.block_size, config.block_size):
                        yield torch.tensor(ids[i : i + config.block_size], dtype=torch.long)

        return TokenIterableDataset()

    tokenized = []
    for text in text_iter:
        ids = tok.encode(text)
        for i in range(0, len(ids) - config.block_size, config.block_size):
            chunk = ids[i : i + config.block_size]
            tokenized.append(torch.tensor(chunk, dtype=torch.long))

    return torch.utils.data.TensorDataset(torch.stack(tokenized))
