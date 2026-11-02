"""Training loop with strictly separated train / validation / test sets."""
from dataclasses import dataclass, asdict
from pathlib import Path
import random
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import make_dataset, ContextData
from .model import ContextTransformer, ModelConfig
from .metrics import classification_metrics


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 250
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    train_size: int = 6000
    val_size: int = 1200
    test_size: int = 2000
    eval_every: int = 25
    seq_len: int = 12
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, dataset: ContextData, *, batch_size=512, device="cpu"):
    model.eval()
    scores = []
    for start in range(0, len(dataset), batch_size):
        idx = slice(start, start + batch_size)
        logits = model(dataset.tokens[idx].to(device), dataset.query_positions[idx].to(device))
        scores.append(logits.argmax(-1).cpu())
    return torch.cat(scores)


def train_one(task: str, mode: str, seed: int, *, training: TrainConfig,
              architecture: ModelConfig, output_dir: Path, save_checkpoint=True) -> dict:
    set_seed(seed)
    device = torch.device(training.device)
    # The SAME datasets across the two modes for a task/seed; no test reuse.
    train_data = make_dataset(training.train_size, task, seed=seed * 101 + 1,
                              seq_len=training.seq_len)
    val_data = make_dataset(training.val_size, task, seed=seed * 101 + 2,
                            seq_len=training.seq_len)
    test_data = make_dataset(training.test_size, task, seed=seed * 101 + 3,
                             seq_len=training.seq_len)
    model = ContextTransformer(architecture, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    generator = torch.Generator().manual_seed(seed * 101 + 4)
    best_val = -1.0
    best_state = None
    history = []
    started = time.perf_counter()
    for step in range(1, training.steps + 1):
        model.train()
        idx = torch.randint(len(train_data), (training.batch_size,), generator=generator)
        logits = model(train_data.tokens[idx].to(device), train_data.query_positions[idx].to(device))
        loss = F.cross_entropy(logits, train_data.targets[idx].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % training.eval_every == 0 or step == training.steps:
            pred_val = predict(model, val_data, device=device)
            val_accuracy = classification_metrics(val_data.targets, pred_val)["accuracy"]
            history.append({"step": step, "train_loss": round(float(loss.item()), 6),
                            "val_accuracy": round(val_accuracy, 6)})
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    test_predictions = predict(model, test_data, device=device)
    test_metrics = classification_metrics(test_data.targets, test_predictions)
    record = {"task": task, "mode": mode, "seed": seed,
              "config": asdict(training), "model": model.metadata(),
              "best_val_accuracy": float(best_val), "test": test_metrics,
              "history": history, "runtime_seconds": round(time.perf_counter() - started, 2),
              "test_targets": test_data.targets.tolist(),
              "test_predictions": test_predictions.tolist()}
    if save_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "mode": mode, "task": task, "seed": seed,
                    "architecture": asdict(architecture), "training": asdict(training)},
                   output_dir / f"{task}_{mode}_seed{seed}.pt")
    return record


def save_results(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": results}, indent=2), encoding="utf8")
