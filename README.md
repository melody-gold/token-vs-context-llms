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
│   ├── io.py: artifact and metric JSON read/write helpers
│   ├── metrics.py: cosine similarity, MSE, and R^2
│   ├── probe.py: affine probe fitting and layerwise evaluation
│   └── summary.py: Markdown summaries for generated layerwise metrics
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
- Markdown summaries of layerwise metric JSON files
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
uv sync --dev --no-editable
uv run pytest
```

Install optional extraction dependencies as well:

```bash
uv sync --all-extras --dev --no-editable
```

Run the debug pipeline:

```bash
uv run token-vs-context extract --config configs/small_debug.yaml
uv run token-vs-context probe --config configs/small_debug.yaml
uv run token-vs-context summarize \
  --metrics results/generated/small_debug_metrics.json \
  --output results/generated/small_debug_summary.md \
  --title "Small Debug Metrics"
uv run token-vs-context plot \
  --metrics results/generated/small_debug_metrics.json \
  --output results/generated/small_debug_metrics.png \
  --title "Small Debug Metrics"
uv run token-vs-context diagnose \
  --config configs/small_debug.yaml \
  --output-dir results/generated/small_debug_diagnostics \
  --title "Small Debug Diagnostics"
```

The extraction step writes a token-level artifact under `artifacts/`. The probe step reads that artifact, fits one linear probe per layer, and writes metrics to `results/`.
The summary step turns the metrics JSON into a compact Markdown table for writeups
or experiment logs. The plot step writes a PNG figure with MSE, `R^2`, and mean
cosine similarity by layer.
The diagnose step writes optional exploratory figures and tables, including
per-token cosine distributions, error by token position, probe-vs-baseline MSE,
activation norm versus error, sampled probe predictions versus target activation
values, a layer-by-position cosine heatmap, and a worst-token table. The
heatmaps aggregate held-out tokens by token position, so they summarize the test
split rather than one selected sequence.

By default, `probe` fits the simple affine baseline with no ridge penalty. Use `--alpha`
or `probe.ridge_alpha` only for later regularized ablations.

If `uv run` is unstable on your machine, the environment created by `uv sync` still works:

```bash
./.venv/bin/pytest
./.venv/bin/token-vs-context probe --config configs/small_debug.yaml
```

If `token-vs-context` fails with `ModuleNotFoundError: No module named
'token_vs_context_llms'`, refresh the environment with uv's non-editable install
mode and then rerun the normal commands:

```bash
uv sync --all-extras --dev --no-editable
```

## Experiment Workflow

1. Start with `configs/small_debug.yaml` to confirm the extraction and probe paths work locally.
2. Use `configs/pythia_pile10k.yaml` for the fast report-scale baseline that
   keeps artifacts manageable.
3. Use one of the larger configs when you want stronger figures without
   overwriting the fast baseline:
   - `configs/pythia_pile10k_70m_100k.yaml`: same Pythia-70M model, larger
     100k-token sample.
   - `configs/pythia_pile10k_160m.yaml`: larger Pythia-160M model, all layers,
     50k tokens.
4. Save the resulting layerwise metrics under `results/` and record the run in
   `docs/final_report/final_report.tex`.
5. Once the hidden-state baseline is stable, extend the artifact format or add a
   parallel path for SAE features.

## Tests and Diagnostic Plot Commands

Run the unit tests by themselves:

```bash
uv run pytest
```

Diagnostics generate the exploratory plot set under `docs/final_report/figures/`.
Run `extract` first if the artifact named in the config does not already exist.

DistilGPT-2 smoke test:

```bash
CONFIG=configs/small_debug.yaml
NAME=small_debug

uv run --no-sync token-vs-context extract --config "$CONFIG"
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

DistilGPT-2 tests plus diagnostics together:

```bash
CONFIG=configs/small_debug.yaml
NAME=small_debug

uv run pytest && \
uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M debug run:

```bash
CONFIG=configs/pythia_debug.yaml
NAME=pythia_debug

uv run --no-sync token-vs-context extract --config "$CONFIG"
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M debug tests plus diagnostics together:

```bash
CONFIG=configs/pythia_debug.yaml
NAME=pythia_debug

uv run pytest && \
uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M Pile-10k report run:

```bash
CONFIG=configs/pythia_pile10k.yaml
NAME=pythia_pile10k

uv run --no-sync token-vs-context extract --config "$CONFIG"
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M Pile-10k tests plus diagnostics together:

```bash
CONFIG=configs/pythia_pile10k.yaml
NAME=pythia_pile10k

uv run pytest && \
uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M 100k-token run:

```bash
CONFIG=configs/pythia_pile10k_70m_100k.yaml
NAME=pythia_70m_pile10k_100k

uv run --no-sync token-vs-context extract --config "$CONFIG"
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-70M 100k-token tests plus diagnostics together:

```bash
CONFIG=configs/pythia_pile10k_70m_100k.yaml
NAME=pythia_70m_pile10k_100k

uv run pytest && \
uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-160M all-layer run:

```bash
CONFIG=configs/pythia_pile10k_160m.yaml
NAME=pythia_160m_pile10k_50k

uv run --no-sync token-vs-context extract --config "$CONFIG"
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

Pythia-160M tests plus diagnostics together:

```bash
CONFIG=configs/pythia_pile10k_160m.yaml
NAME=pythia_160m_pile10k_50k

uv run pytest && \
uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

The full command pattern for any config is:

```bash
CONFIG=configs/pythia_pile10k_160m.yaml
NAME=pythia_160m_pile10k_50k

uv run --no-sync token-vs-context extract --config "$CONFIG" && \
uv run --no-sync token-vs-context probe --config "$CONFIG" && \
uv run --no-sync token-vs-context summarize \
  --metrics "results/generated/${NAME}_metrics.json" \
  --output "results/generated/${NAME}_summary.md" \
  --title "$NAME Metrics" && \
uv run --no-sync token-vs-context plot \
  --metrics "results/generated/${NAME}_metrics.json" \
  --output "docs/final_report/figures/${NAME}_metrics.png" \
  --title "$NAME Metrics" && \
uv run --no-sync token-vs-context diagnose \
  --config "$CONFIG" \
  --output-dir "docs/final_report/figures/${NAME}_diagnostics" \
  --title "$NAME Diagnostics"
```

## Probe Implementation Sketch

The supervised dataset for each probe follows the cache-tensor workflow:

```python
logits, cache = model.run_with_cache(data)

embed_activations = cache["hook_embed"]              # [batch, n_ctx, d_model]
layer_activations = cache[f"blocks.{n}.hook_resid_post"]  # [batch, n_ctx, d_model]

x = embed_activations.reshape(-1, d_model)
y = layer_activations.reshape(-1, d_model)
```

In this repository, `extract` does the same conceptual step with Hugging Face
hidden states instead of TransformerLens cache names: it stores context-free
input embedding vectors as `token_embeddings` and selected contextual block
outputs as `hidden_states`. Padded positions are removed, so the saved arrays are
already token-level rows.

The saved artifact names map to the sketch as follows:

```python
artifact = np.load("artifacts/extractions/pythia_debug.npz")

embed_activations = artifact["token_embeddings"]      # [num_tokens, d_model]
all_intermediate_activations = artifact["hidden_states"]  # [num_tokens, num_layers, d_model]
layer_indices = artifact["layer_indices"]

layer_position = 0
intermediate_activation = all_intermediate_activations[:, layer_position, :]
model_layer = layer_indices[layer_position]
```

The probe is the affine baseline:

```python
lens = nn.Linear(d_model, d_model, bias=True)
prediction = lens(batch_x)
loss = mse(prediction, batch_y)
```

`src/token_vs_context_llms/probe.py` implements the same linear map. For the
default baseline, it solves the MSE objective directly with least squares rather
than taking gradient steps, then evaluates on a shuffled held-out split. This is
equivalent to training an unregularized `nn.Linear` to convergence on the same
flattened `(n_samples, d_model)` activation pairs. The helper
`flatten_token_activations` accepts either already-flat arrays or cache-shaped
`[batch, n_ctx, d_model]` tensors.

## Notes

- The extraction code is intentionally minimal so the core probe utilities and tests stay lightweight.
- If you use Gemma or another gated model, you may need local Hugging Face authentication before extraction.
- The repo currently supports the hidden-state baseline directly; the SAE stage is planned but not implemented yet.
