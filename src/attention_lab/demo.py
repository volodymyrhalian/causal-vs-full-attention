"""Interactive, explainable command-line demo using included trained checkpoints."""
import argparse
from pathlib import Path
import torch
from .data import make_dataset, token_description
from .model import ContextTransformer, ModelConfig


def main(argv=None):
    p = argparse.ArgumentParser(description="Try a trained context Transformer")
    p.add_argument("--task", choices=["left", "right", "xor"], default="xor")
    p.add_argument("--mode", choices=["causal", "full"], default="full")
    p.add_argument("--example-seed", type=int, default=2026)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args(argv)
    path = args.checkpoint or Path("models") / f"{args.task}_{args.mode}_seed7.pt"
    if not path.is_file():
        p.error(f"Checkpoint not found: {path}. Run in repository root or supply --checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["mode"] != args.mode or state["task"] != args.task:
        p.error("Checkpoint metadata disagrees with requested mode/task")
    model = ContextTransformer(ModelConfig(**state["architecture"]), args.mode)
    model.load_state_dict(state["model"])
    model.eval()
    sample = make_dataset(1, args.task, seed=args.example_seed,
                          seq_len=state["training"]["seq_len"])
    edited = sample.tokens.clone()
    query = sample.query_positions[0].item()
    edited[0, query+1:] = 6
    with torch.no_grad():
        logits = model(sample.tokens, sample.query_positions)
        logits_edited = model(edited, sample.query_positions)
        prob = logits.softmax(dim=-1)[0]
    print(f"Task: {args.task} | Mode: {args.mode} | Query position: {query}")
    print("Tokens:")
    for i, token in enumerate(token_description(sample.tokens[0])):
        print(f"  {i:2d}. {token:<12} {'<-- CLASSIFY HERE' if i == query else ''}")
    print(f"True label: {int(sample.targets[0])} | Prediction: {int(prob.argmax())} "
          f"| P(label=1): {float(prob[1]):.4f}")
    print(f"Largest change in logits after replacing ONLY future tokens: "
          f"{float((logits-logits_edited).abs().max()):.8f}")
    print("Causal should be ~0; full attention can change.")


if __name__ == "__main__":
    main()
