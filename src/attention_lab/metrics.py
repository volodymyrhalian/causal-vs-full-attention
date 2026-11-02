"""Evaluation and paired bootstrap uncertainty for fixed test examples."""
import numpy as np
import torch


def classification_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> dict:
    t = targets.detach().cpu().numpy()
    p = predictions.detach().cpu().numpy()
    count = len(t)
    conf = np.zeros((2, 2), dtype=np.int64)
    for a, b in zip(t, p):
        conf[int(a), int(b)] += 1
    recalls = np.diag(conf) / np.maximum(conf.sum(axis=1), 1)
    return {"accuracy": float((t == p).mean()), "balanced_accuracy": float(recalls.mean()),
            "confusion_matrix": conf.tolist(), "count": count}


def paired_bootstrap_accuracy_diff(targets, prediction_a, prediction_b, *,
                                   seed=1234, samples=2000):
    """CI for full-minus-causal accuracy on the SAME held-out examples.

    Conditional on the trained models; does not quantify variation between
    independent training runs (report multi-seed mean/std for that).
    """
    t = np.asarray(targets)
    a = np.asarray(prediction_a)
    b = np.asarray(prediction_b)
    if not (len(t) == len(a) == len(b)):
        raise ValueError("all arrays must have equal length")
    if len(t) == 0:
        raise ValueError("cannot bootstrap empty sample")
    delta = (b == t).astype(float) - (a == t).astype(float)
    rng = np.random.default_rng(seed)
    # Chunked sampling to limit peak memory for larger test sets.
    boot = []
    for _ in range(0, samples, 100):
        take = min(100, samples - len(boot))
        indices = rng.integers(0, len(t), size=(take, len(t)))
        boot.extend(delta[indices].mean(axis=1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta_full_minus_causal": float(delta.mean()),
            "ci95": [float(lo), float(hi)], "resamples": samples}
