import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
DATA_DIR = Path(__file__).resolve().parent / 'data'

import torch
from model import GPT
from advanced import RewardTextDataset, char_reward_fn


def test_offline_rl_dataset():
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    assert len(ds) == 3
    assert ds.vocab_size > 0


def test_reward_augmented_generation():
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    model = GPT(ds.vocab_size, block_size=16)
    reward_fn = char_reward_fn(['g'], ds.stoi)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    out = model.generate_reward_augmented(idx, max_new_tokens=2,
                                          reward_fn=reward_fn, beta=0.5)
    assert out.shape[1] == 3


def test_generate_basic():
    """Test the standard generate() method."""
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    model = GPT(ds.vocab_size, block_size=16)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 6)  # 1 seed + 5 generated


def test_generate_temperature():
    """Test that temperature=0.001 produces near-deterministic output."""
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    model = GPT(ds.vocab_size, block_size=16)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    torch.manual_seed(42)
    out1 = model.generate(idx.clone(), max_new_tokens=3, temperature=0.001)
    torch.manual_seed(42)
    out2 = model.generate(idx.clone(), max_new_tokens=3, temperature=0.001)
    assert torch.equal(out1, out2)
