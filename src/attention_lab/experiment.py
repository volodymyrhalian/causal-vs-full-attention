"""Experiment CLI: python -m attention_lab.experiment --help"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from .model import ModelConfig
from .train import TrainConfig, train_one, save_results
from .metrics import paired_bootstrap_accuracy_diff
from .plots import plot_comparison, plot_learning, plot_masks, plot_checkpoint_attention


def main(argv=None):
    parser = argparse.ArgumentParser(description="Leakage-controlled causal vs full attention study")
    parser.add_argument("--tasks", nargs="+", choices=["left", "right", "xor"], default=["left", "right", "xor"])
    parser.add_argument("--modes", nargs="+", choices=["causal", "full"], default=["causal", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/full_benchmark"))
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.eval_every < 1 or not args.seeds:
        parser.error("steps, eval-every, and seeds must be positive")
    torch.set_num_threads(args.threads)
    arch = ModelConfig(d_model=args.d_model, num_layers=args.layers, num_heads=args.heads,
                       d_ff=args.d_model * 2)
    config = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                         train_size=args.train_size, val_size=args.val_size,
                         test_size=args.test_size, eval_every=args.eval_every,
                         device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for task in args.tasks:
        for seed in args.seeds:
            for mode in args.modes:
                print(f"Training task={task:<5} mode={mode:<6} seed={seed} ...", flush=True)
                rec = train_one(task, mode, seed, training=config, architecture=arch,
                                output_dir=args.output / "checkpoints",
                                save_checkpoint=args.save_checkpoints)
                runs.append(rec)
                print(f"  val={rec['best_val_accuracy']:.3f} test={rec['test']['accuracy']:.3f} "
                      f"({rec['runtime_seconds']:.1f}s)", flush=True)
    save_results(runs, args.output / "runs.json")
    # Flat compact CSV for comparing conditions and seeds.
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "mode", "seed", "parameters", "validation_accuracy",
                                             "test_accuracy", "test_balanced_accuracy", "runtime_seconds"])
        w.writeheader()
        for r in runs:
            w.writerow({"task": r["task"], "mode": r["mode"], "seed": r["seed"],
                        "parameters": r["model"]["parameter_count"],
                        "validation_accuracy": r["best_val_accuracy"],
                        "test_accuracy": r["test"]["accuracy"],
                        "test_balanced_accuracy": r["test"]["balanced_accuracy"],
                        "runtime_seconds": r["runtime_seconds"]})
    comparisons = []
    for task in args.tasks:
        for seed in args.seeds:
            c = next((r for r in runs if r["task"] == task and r["seed"] == seed and r["mode"] == "causal"), None)
            f = next((r for r in runs if r["task"] == task and r["seed"] == seed and r["mode"] == "full"), None)
            if c and f:
                results = paired_bootstrap_accuracy_diff(c["test_targets"], c["test_predictions"],
                             f["test_predictions"], seed=seed + 10000)
                comparisons.append({"task": task, "seed": seed, **results})
    (args.output / "paired_bootstrap.json").write_text(json.dumps(comparisons, indent=2), encoding="utf8")
    plot_masks(args.output / "attention_masks.png", length=config.seq_len)
    plot_comparison(runs, args.output / "test_accuracy.png")
    plot_learning(runs, args.output / "learning_curves.png")
    if args.save_checkpoints:
        for task in args.tasks:
            for mode in args.modes:
                ckpt = args.output / "checkpoints" / f"{task}_{mode}_seed{args.seeds[0]}.pt"
                plot_checkpoint_attention(ckpt,
                    args.output / f"attention_{task}_{mode}.png")
    (args.output / "environment.json").write_text(json.dumps({
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "system": platform.platform(), "torch_num_threads": torch.get_num_threads(),
        "command": vars(args) | {"output": str(args.output)}}, indent=2, default=str), encoding="utf8")
    print(f"\nSaved metrics, plots, and reproducibility metadata in: {args.output}")


if __name__ == "__main__":
    main()
