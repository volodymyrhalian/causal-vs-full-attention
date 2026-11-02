# Methodology and research protocol

## Question

What does access to future tokens change in a Transformer when **every other
modeling decision is kept fixed**? This experiment is about *context visibility
for classification at a fixed query position*, not about next-token language
modeling or general natural-language understanding.

## Hypothesis

A causal classifier reading only the hidden state at position `q` cannot use
information located after `q`. A full-attention classifier can. If the target
is determined by an independent right-side bit, the optimal causal accuracy
is 50% (assuming balanced bits), even if the model has unlimited capacity.
The full-attention architecture can solve the task.

## Data generation

Sequences consist of 12 discrete tokens and one distinguished `<QUERY>` token
at a random position in `[3, 8]`. A `LEFT:0` / `LEFT:1` clue is placed 1-3
positions before `<QUERY>`, and a `RIGHT:0` / `RIGHT:1` clue 1-3 positions
after it. Remaining tokens are IID distractors, unrelated to both bits.

Left and right bits are independently sampled as Bernoulli(0.5). The target
is one of:

| Task | Target | Bayes-optimal causal at query | Full attention |
|---|---|---|---|
| `left` | left bit | 100% | 100% |
| `right` | right bit | 50% | 100% |
| `xor` | left XOR right | 50% | 100% |

Theoretical rates are population limits. Finite test-set observations fluctuate.

**Important:** The label is **never included** as an input token. The query
position is specified, and both models classify **only that position**. If we
pooled all token states, even a causally masked encoder could see information
from later positions; that would undermine this experiment. The code prevents
that information leak explicitly.

## Controlled architecture

- Two pre-LayerNorm residual Transformer blocks
- Learned token + positional embeddings
- Four attention heads, `d_model=48`, FFN hidden dimension 96
- Final query-position pooling, linear two-class head
- **40,034 trainable parameters** per model, independent of attention mode
- PyTorch's scaled dot-product attention; `is_causal=True` vs `False`

All datasets, initial weights (given seed), optimizer, learning rate, epoch
budget, model checkpoint selection, batch order, and evaluation are shared
between modes. Only attention visibility changes.

## Training and validation

- `6,000` training sequences; `1,200` validation; `2,000` held-out test;
  synthetic but disjoint RNG streams per split
- `360` AdamW steps per run, batch 128, learning rate `0.002`, weight decay `0.01`
- No dropout; cross-entropy, gradient norm clipping (1.0)
- Best validation-accuracy checkpoint selected every 40 optimizer steps
- Independent training seeds `7`, `11`, `19`, six task/mode combinations each
- Each pair uses identical test examples; final accuracy uses only the selected
  validation checkpoint; we do not tune individual models on the test set

This is an **exploratory synthetic study**, not a pre-registered benchmark.
The step count and architecture were developed using pilot runs, so general
claims about model selection on a fully blind external test set are unwarranted.

## Statistical uncertainty

The `aggregate.json` file includes mean and sample standard deviation across
three training seeds. `paired_bootstrap.json` gives 95% percentile intervals
for full-minus-causal test accuracy, using 2,000 paired resamples for **each**
seed/task. Because pairs share examples, paired resampling is more appropriate
than independently resampling mode accuracies. These intervals are conditional
on trained models and **do not quantify retraining variation**; report both.

## Auditing

- `tests/test_attention.py::test_causal_future_invariance` actually replaces
  all future tokens and verifies causal predictions do not change.
- `tests/test_attention.py::test_future_weights_are_zero` verifies masked
  attention probabilities for future keys are exactly zero.
- `tests/test_attention.py::test_parameter_matching` ensures equal capacity.
- `tests/test_data.py` verifies correct marker ordering, targets and seeds.
- `scripts/verify_results.py` recomputes accuracies from stored raw predictions.
- Only the small synthetic dataset generator is necessary for reproduction.

## Limitations

1. Synthetic patterns are deliberately simple; high accuracy must **not** be
   interpreted as real-world NLP capability or production readiness.
2. A causal model could use the right-side bit if the query were moved to the
   end of the sequence; the constraint here is the position of the readout.
3. The two architectures have equal parameter counts, not necessarily equal
   compute cost or optimized wall-clock execution paths.
4. Attention maps are informative diagnostics, not proof of causal feature
   attribution or faithful model explanation.
5. Three seeds measure some training variation; larger studies with natural
   datasets, more seeds and independent test sets are needed for broader claims.
6. Right-context cases are **information-theoretically impossible** for the
   specified causal readout, not evidence that one Transformer family is
   universally superior.

## Reference reading

- Vaswani et al. (2017), *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*: https://arxiv.org/abs/1810.04805
- PyTorch SDPA API and mask semantics: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
