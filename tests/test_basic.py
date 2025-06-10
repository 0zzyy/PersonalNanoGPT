import torch

from src.model import GPT, GPTConfig
from src.tokenizer import Tokenizer


def test_model_forward() -> None:
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (1, config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (1, config.block_size, config.vocab_size)
    assert loss is not None


def test_tokenizer_roundtrip() -> None:
    tok = Tokenizer()
    text = "Hello world"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert isinstance(ids, list)
    assert out.strip() == text
