import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
DATA_DIR = Path(__file__).resolve().parent / 'data'

import torch
from model import GPT
from advanced import RewardTextDataset, char_reward_fn

# simple stress scenarios on small model

def run_generation(domain_chars):
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    model = GPT(ds.vocab_size, block_size=16)
    reward_fn = char_reward_fn(domain_chars, ds.stoi)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    out = model.generate_reward_augmented(idx, max_new_tokens=3,
                                          reward_fn=reward_fn, beta=0.5)
    return out


def test_stress_cases():
    scenarios = [
        ['g'],  # normal domain token
        ['z'],  # unseen reward character
        [],      # no reward guidance
        ['h', 'e', 'l'],
        ['o'],
        ['w'],
        ['x'],  # conflicting signals
        ['g', 'w'],
        ['m'],
        ['d'],
    ]
    for chars in scenarios:
        out = run_generation(chars)
        assert out.shape[1] == 4


def test_model_with_dropout():
    """Verify model works with dropout enabled."""
    ds = RewardTextDataset(DATA_DIR / 'offline_rl.txt', block_size=16)
    model = GPT(ds.vocab_size, block_size=16, dropout=0.1)
    idx = torch.tensor([[ds.stoi['h']]], dtype=torch.long)
    # Training mode: dropout active
    model.train()
    logits = model(idx)
    assert logits.shape == (1, 1, ds.vocab_size)
    # Eval mode: dropout inactive, generation works
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape == (1, 3)
