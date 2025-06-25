import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from explore import KnowledgeDataset
from model import GPT
from advanced import char_reward_fn


def run_case(filename, reward_chars):
    path = Path('tests/data') / filename
    ds = KnowledgeDataset(path, block_size=8)
    model = GPT(ds.vocab_size, block_size=8)
    if ds.vocab_size == 0:
        return
    reward_fn = char_reward_fn(reward_chars, ds.stoi)
    first_token = next(iter(ds.stoi.values()))
    idx = torch.tensor([[first_token]], dtype=torch.long)
    out = model.generate_reward_augmented(idx, max_new_tokens=2,
                                          reward_fn=reward_fn, beta=0.5)
    assert out.shape[1] == 3


def test_knowledge_dataset_variants():
    scenarios = [
        ('sample.txt', ['t']),
        ('sample.txt', []),
        ('sample.csv', ['a']),
        ('sample.csv', ['1']),
        ('sample.csv', []),
        ('sample.jsonl', ['h']),
        ('sample.jsonl', ['0']),
        ('sample.jsonl', []),
        ('sample.txt', ['u']),
        ('sample.csv', ['s']),
    ]
    for fname, chars in scenarios:
        run_case(fname, chars)
