# Token vs Context in LLM Representations

Repository for the MATH 598 final project on measuring how much of an LLM's hidden representation is recoverable from the current token alone versus prior context.

The core project question is:

Can an intermediate representation at layer `l` be predicted from the current token embedding alone, without access to earlier tokens?

The updated proposal breaks the work into two stages:

1. Run a hidden-state baseline: extract token embeddings and per-layer activations from a pretrained causal language model, then fit one token-only linear probe per layer.
2. Extend the same framework to sparse autoencoder (SAE) features to ask which interpretable features are token-local versus context-dependent.

## Repository Layout & Code Map

```
token-vs-context-llms/
├── writeup/
│   ├── progress_report.md: update and experiment log scaffold
│   ├── proposal/
│   │   ├── proposal.tex: proposal source
│   │   ├── proposal.pdf: compiled proposal
│   │   └── sources.bib: bibliography for the proposal
│
├── src/token_vs_context_llms/: extraction, probing, metrics, and CLI code
│   ├── cli.py: command-line entrypoint exposed as `token-vs-context`
│   ├── config.py: experiment dataclasses and YAML loader
│   ├── extract.py: Hugging Face extraction pipeline for token embeddings and hidden states
│   ├── metrics.py: cosine similarity, MSE, and R^2
│   └── probe.py: affine probe fitting and layerwise evaluation
│
├── tests/: lightweight unit tests for metrics and probe fitting
├── configs/: runnable configs for smoke tests plus templates for larger experiments
├── data/: dataset notes, manifests, and local subsets
├── artifacts/: local token-level activations and future SAE-derived artifacts
└── results/: generated layerwise metrics, plots, and experiment summaries
```

The `data/`, `artifacts/`, and `results/` directories are currently scaffold directories.
They intentionally contain only README files until experiments are run locally.
Large activation artifacts should not be committed because they are regenerable.


## Current Scope

Implemented now:

- token embedding and hidden-state extraction from a Hugging Face causal LM
- uncompressed artifact storage with sidecar metadata
- unregularized affine probes trained separately for each selected layer
- evaluation with cosine similarity, mean squared error, and `R^2`
- a small local debug config for validating the pipeline end to end
- no proposal-scale results are committed yet

Planned next:

- choose the main model and dataset for the proposal-scale run
- collect and commit layerwise metric summaries at a larger token budget
- add plotting and experiment summaries
- add an SAE activation path once a model-compatible SAE release is selected


## Quickstart

Install the lightweight environment and run tests:

```bash
uv sync --dev
uv run pytest
```

Install optional extraction dependencies as well:

```bash
uv sync --all-extras --dev
```

Run the debug pipeline:

```bash
uv run token-vs-context extract --config configs/small_debug.yaml
uv run token-vs-context probe --config configs/small_debug.yaml
```

The extraction step writes a token-level artifact under `artifacts/`. The probe step reads that artifact, fits one linear probe per layer, and writes metrics to `results/`.

By default, `probe` fits the simple affine baseline with no ridge penalty. Use `--alpha`
or `probe.ridge_alpha` only for later regularized ablations.

If `uv run` is unstable on your machine, the environment created by `uv sync` still works:

```bash
./.venv/bin/pytest
./.venv/bin/token-vs-context probe --config configs/small_debug.yaml
```

## Experiment Workflow

1. Start with `configs/small_debug.yaml` to confirm the extraction and probe paths work locally.
2. Copy `configs/experiment_template.yaml` and replace the model, dataset, and output paths for the main run.
3. Save the resulting layerwise metrics under `results/` and record the run in `writeup/progress_report.md`.
4. Once the hidden-state baseline is stable, extend the artifact format or add a parallel path for SAE features.

## Notes

- The extraction code is intentionally minimal so the core probe utilities and tests stay lightweight.
- If you use Gemma or another gated model, you may need local Hugging Face authentication before extraction.
- The repo currently supports the hidden-state baseline directly; the SAE stage is planned but not implemented yet.
