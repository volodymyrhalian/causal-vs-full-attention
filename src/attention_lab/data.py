"""Deterministic, leakage-auditable token-level context tasks.

Sequence format (length 12 by default):
- one QUERY at a randomly chosen center position;
- one LEFT_0/LEFT_1 marker before the query;
- one RIGHT_0/RIGHT_1 marker after the query;
- all other tokens are independent distractors.

Targets are the left bit, right bit, or their XOR. The right bit is statistically
independent of all visible prefix tokens, so a strictly causal query-position
classifier cannot exceed ~50% accuracy on RIGHT or XOR in expectation.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch

Task = Literal["left", "right", "xor"]
PAD, QUERY, LEFT_0, LEFT_1, RIGHT_0, RIGHT_1 = range(6)
VOCAB_SIZE = 24
TOKEN_NAMES = {0: "<PAD>", 1: "<QUERY>", 2: "LEFT:0", 3: "LEFT:1", 4: "RIGHT:0", 5: "RIGHT:1"}

@dataclass(frozen=True)
class ContextData:
    tokens: torch.LongTensor
    query_positions: torch.LongTensor
    targets: torch.LongTensor
    left_bits: torch.LongTensor
    right_bits: torch.LongTensor

    def __len__(self) -> int:
        return int(self.tokens.size(0))

    def subset(self, indices: torch.Tensor) -> "ContextData":
        return ContextData(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))


def make_dataset(n: int, task: Task, *, seed: int, seq_len: int = 12) -> ContextData:
    if task not in ("left", "right", "xor"):
        raise ValueError(f"unknown task: {task}")
    if n <= 0 or seq_len < 9:
        raise ValueError("n must be positive and seq_len must be at least 9")
    rng = np.random.default_rng(seed)
    # All bits are independently sampled; the label is never stored in tokens.
    left_bits = rng.integers(0, 2, n, dtype=np.int64)
    right_bits = rng.integers(0, 2, n, dtype=np.int64)
    positions = rng.integers(3, seq_len - 3, n, dtype=np.int64)
    left_offsets = rng.integers(1, 4, n, dtype=np.int64)
    right_offsets = rng.integers(1, 4, n, dtype=np.int64)
    tokens = rng.integers(6, VOCAB_SIZE, size=(n, seq_len), dtype=np.int64)
    idx = np.arange(n)
    tokens[idx, positions] = QUERY
    tokens[idx, positions - left_offsets] = LEFT_0 + left_bits
    tokens[idx, positions + right_offsets] = RIGHT_0 + right_bits
    if task == "left":
        targets = left_bits
    elif task == "right":
        targets = right_bits
    else:
        targets = left_bits ^ right_bits
    return ContextData(
        tokens=torch.from_numpy(tokens),
        query_positions=torch.from_numpy(positions),
        targets=torch.from_numpy(targets.copy()),
        left_bits=torch.from_numpy(left_bits),
        right_bits=torch.from_numpy(right_bits),
    )


def token_description(tokens: torch.Tensor) -> list[str]:
    return [TOKEN_NAMES.get(int(t), f"noise:{int(t)}") for t in tokens]
