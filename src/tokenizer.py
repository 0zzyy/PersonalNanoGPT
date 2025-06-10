"""Wrapper around the GPT-2 byte pair tokenizer."""

from __future__ import annotations

import tiktoken


class Tokenizer:
    """Simple wrapper for the GPT‑2 tokenizer from ``tiktoken``."""

    def __init__(self) -> None:
        self._tok = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
