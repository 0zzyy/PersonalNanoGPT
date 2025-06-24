import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from model import GPT
from advanced import RewardTextDataset, char_reward_fn

# simple stress scenarios on small model

def run_generation(domain_chars):
    ds = RewardTextDataset(Path('tests/data/offline_rl.txt'), block_size=16)
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


