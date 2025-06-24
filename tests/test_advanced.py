import torch
from pathlib import Path

from model import GPT
from advanced import RewardTextDataset, char_reward_fn


def test_offline_rl_dataset():
    ds = RewardTextDataset(Path('tests/data/offline_rl.txt'), block_size=16)
    assert len(ds) == 3
    assert ds.vocab_size > 0


def test_reward_augmented_generation():
    ds = RewardTextDataset(Path('tests/data/offline_rl.txt'), block_size=16)
    model = GPT(ds.vocab_size, block_size=16)
    reward_fn = char_reward_fn(['g'], ds.stoi)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    out = model.generate_reward_augmented(idx, max_new_tokens=2,
                                          reward_fn=reward_fn, beta=0.5)
    assert out.shape[1] == 3

