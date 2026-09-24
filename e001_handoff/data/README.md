# E001 data manifests

The JSONL files in `fit`, `validation`, `locked`, and `long` contain only the
exact token IDs needed by E001 plus immutable source metadata and hashes. They
are generated from the corpus commits, files, candidate universes, tokenizer,
and hash-order rule frozen in `PREREGISTRATION.json`.

`long/manifest.jsonl` is selected and isolated before evaluation, but it may be
read by model-scoring code only if the preregistered 4K verdict gate fires.

Raw source text remains in the verified Hugging Face cache and is not copied
into the repository. No model score, perplexity, agreement, state error, or
handoff result participates in document selection.
