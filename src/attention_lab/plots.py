"""Self-contained visualizations produced from real experiment outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import attention_visibility


def plot_masks(path: Path, length=12, query=5):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, mode in zip(axes, ("causal", "full")):
        visible = attention_visibility(length, mode == "causal").numpy()
        ax.imshow(visible, vmin=0, vmax=1, cmap="Blues")
        ax.axhline(query, color="tab:red", alpha=0.6, linewidth=1)
        ax.axvline(query, color="tab:red", alpha=0.6, linewidth=1)
        ax.set(title=f"{mode.title()} attention", xlabel="Key position", ylabel="Query position")
        ax.set_xticks(range(0, length, 2))
        ax.set_yticks(range(0, length, 2))
    fig.suptitle("Attention visibility (blue = allowed)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(runs: list[dict], path: Path):
    tasks = [t for t in ("left", "right", "xor") if t in {r["task"] for r in runs}]
    modes = ("causal", "full")
    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for idx, mode in enumerate(modes):
        means, stds = [], []
        for task in tasks:
            scores = [r["test"]["accuracy"] for r in runs if r["task"] == task and r["mode"] == mode]
            means.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores, ddof=1) if len(scores) > 1 else 0)
        if any(r["mode"] == mode for r in runs):
            ax.bar(x + (idx - 0.5) * width, means, width=width, label=mode.title(),
                   yerr=stds, capsize=3)
    ax.axhline(0.5, linestyle="--", linewidth=1, label="chance (binary)")
    ax.set_xticks(x, ["Left-context", "Right-context", "XOR"][:len(tasks)])
    if len(tasks) != 3:
        ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylim(0, 1.07)
    ax.set_ylabel("Held-out test accuracy")
    ax.set_title("Effect of attention visibility on context reasoning")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_learning(runs: list[dict], path: Path):
    tasks = [t for t in ("left", "right", "xor") if t in {r["task"] for r in runs}]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, task in zip(axes[0], tasks):
        for mode in ("causal", "full"):
            items = [r for r in runs if r["task"] == task and r["mode"] == mode]
            if not items:
                continue
            # Plot median across seeds on aligned evaluation steps.
            steps = [h["step"] for h in items[0]["history"]]
            values = np.array([[h["val_accuracy"] for h in r["history"]] for r in items])
            ax.plot(steps, np.median(values, axis=0), marker="o", markersize=3, label=mode)
        ax.axhline(.5, linestyle="--", linewidth=1, label="chance")
        ax.set(title=task.upper(), xlabel="Optimizer steps", ylim=(0, 1.02))
        ax.legend()
    axes[0][0].set_ylabel("Validation accuracy")
    fig.suptitle("Learning curves (never based on the test set)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_checkpoint_attention(checkpoint: Path, path: Path, *, example_seed=2026):
    import torch
    from .data import make_dataset, token_description
    from .model import ContextTransformer, ModelConfig
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ContextTransformer(ModelConfig(**snapshot["architecture"]), snapshot["mode"])
    model.load_state_dict(snapshot["model"])
    model.eval()
    data = make_dataset(1, snapshot["task"], seed=example_seed,
                        seq_len=snapshot["training"]["seq_len"])
    with torch.no_grad():
        logits, maps = model(data.tokens, data.query_positions, inspect=True)
        pred = logits.argmax(dim=-1).item()
    q = int(data.query_positions[0].item())
    # Average attention heads of final layer as one descriptive visualization.
    weights = maps[-1].squeeze(0).mean(dim=0).cpu().numpy()
    names = token_description(data.tokens[0])
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    im = ax.imshow(weights, vmin=0, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.axhline(q, color="tab:red", linewidth=1)
    ax.axvline(q, color="tab:red", linewidth=1)
    ax.set(title=f"{snapshot['task']} · {snapshot['mode']} · layer {len(maps)} · mean over heads\n"
                 f"query index {q} | predicted {pred} | true {int(data.targets[0])}",
           xlabel="Key", ylabel="Query")
    fig.colorbar(im, ax=ax, fraction=.045, pad=.02, label="Attention probability")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155)
    plt.close(fig)
