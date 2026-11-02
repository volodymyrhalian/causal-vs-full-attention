# Model card: Query-position context Transformers

**Type:** Research / educational controlled benchmark

**Model families:** Parameter-matched two-layer Transformers with causal vs
unrestricted self-attention. Each has 40,034 parameters and predicts one of
two synthetic labels for a designated query token.

**Intended usage:** Unit testing mask behavior, understanding sequence
classification information flow, prototyping attention ablations, and
reproducible educational experiments. Included checkpoints use seed 7.

**Out of scope:** Real-world sentiment analysis, audio recognition, high-stakes
decisions, privacy-sensitive inference or any domain where these toy tokens
bear no semantic relationship to the intended labels.

**Training data:** Fully synthetic sequences generated in `data.py`, with two
independent clue bits and independent noise tokens. There are no human-subject
recordings, personal information, or external dataset terms to download.

**Evaluation:** Three tasks; paired training modes on matched splits across
three seeds. See `results/benchmark_3seeds` for metrics and prediction vectors.

**Bias / limitations:** Artificial data is balanced by design in expectation.
The comparison tests one visibility restriction; it does not establish
performance differences for general language models or arbitrary training
objectives. The full-attention model has an information advantage by design.

**Provenance:** Independent portfolio demonstration created as a new
implementation of a research concept; it is **not** a recovered or previously
submitted university project, and it should not be presented as one.
