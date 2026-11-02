import pytest
import torch
from attention_lab.data import make_dataset, QUERY, LEFT_0, RIGHT_0

@pytest.mark.parametrize("task", ["left", "right", "xor"])
def test_labels_are_from_independent_cues(task):
    data = make_dataset(300, task, seed=99)
    for i in range(len(data)):
        q = data.query_positions[i].item()
        tokens = data.tokens[i].tolist()
        assert tokens[q] == QUERY
        left_idx = [j for j,t in enumerate(tokens) if t in [2,3]]
        right_idx = [j for j,t in enumerate(tokens) if t in [4,5]]
        assert len(left_idx) == len(right_idx) == 1
        assert left_idx[0] < q < right_idx[0]
        left, right = tokens[left_idx[0]] - LEFT_0, tokens[right_idx[0]] - RIGHT_0
        expected = {"left": left, "right": right, "xor": left ^ right}[task]
        assert int(data.targets[i]) == expected


def test_repeatable_and_independent_splits():
    a = make_dataset(100, "xor", seed=42)
    b = make_dataset(100, "xor", seed=42)
    c = make_dataset(100, "xor", seed=43)
    assert torch.equal(a.tokens, b.tokens)
    assert not torch.equal(a.tokens, c.tokens)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_dataset(10, "not_a_task", seed=1)
    with pytest.raises(ValueError):
        make_dataset(0, "left", seed=1)
