"""Check the repository's recorded metrics against per-example predictions."""
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark_3seeds/runs.json")
runs = json.loads(path.read_text(encoding="utf8"))["runs"]
assert len(runs) > 0
by_group = defaultdict(list)
for run in runs:
    correct = sum(a == b for a, b in zip(run["test_targets"], run["test_predictions"]))
    recomputed = correct / len(run["test_targets"])
    assert math.isclose(recomputed, run["test"]["accuracy"], abs_tol=1e-10)
    by_group[(run["task"], run["mode"])].append(recomputed)
print(f"Verified {len(runs)} runs against raw predictions")
for (task, mode), accuracies in sorted(by_group.items()):
    print(f"{task:>5} {mode:>6}: mean {sum(accuracies)/len(accuracies):.4f}")
