# Token vs Context in LLM Representations

Repository for the MATH 598 final project on measuring how much of an LLM's hidden representation is recoverable from the current token alone versus prior context.

The core project question is:

Can an intermediate representation at layer `l` be predicted from the current token embedding alone, without access to earlier tokens?

The updated proposal breaks the work into two stages:

1. Run a hidden-state baseline: extract token embeddings and per-layer activations from a pretrained causal language model, then fit one token-only linear probe per layer.
2. Extend the same framework to sparse autoencoder (SAE) features to ask which interpretable features are token-local versus context-dependent.

## Repository Layout & Code Map




## Current Scope

Implemented now:

- token embedding and hidden-state extraction from a Hugging Face causal LM
- compressed artifact storage with sidecar metadata
- ridge-style linear probes trained separately for each selected layer
- evaluation with cosine similarity, mean squared error, and `R^2`
- a small local debug config for validating the pipeline end to end

Planned next:

- choose the main model and dataset for the proposal-scale run
- collect layerwise results at a larger token budget
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
