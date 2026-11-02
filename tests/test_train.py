import json
from pathlib import Path
import torch
from attention_lab.metrics import classification_metrics, paired_bootstrap_accuracy_diff
from attention_lab.model import ModelConfig
from attention_lab.train import TrainConfig, train_one


def test_metrics_and_bootstrap():
    t = torch.tensor([0, 0, 1, 1])
    a = torch.tensor([0, 1, 0, 1])
    b = torch.tensor([0, 0, 1, 1])
    assert classification_metrics(t, a)["accuracy"] == .5
    out = paired_bootstrap_accuracy_diff(t.tolist(), a.tolist(), b.tolist(), samples=500)
    assert out["delta_full_minus_causal"] == .5
    assert out["ci95"][1] <= 1.0


def test_end_to_end_smoke(tmp_path):
    cfg = TrainConfig(steps=3, eval_every=1, batch_size=16,
                      train_size=48, val_size=20, test_size=20)
    rec = train_one("right", "causal", 7, training=cfg,
                    architecture=ModelConfig(d_model=16, num_heads=2, d_ff=32, num_layers=1),
                    output_dir=tmp_path, save_checkpoint=True)
    assert 0 <= rec["test"]["accuracy"] <= 1
    assert (tmp_path / "right_causal_seed7.pt").is_file()
