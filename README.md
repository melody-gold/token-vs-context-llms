# Token vs Context in LLM Representations

Minimal starter repository for the MATH 598 final project on separating token-local information from context-dependent information in transformer representations.

The current project question is:

Can an intermediate LLM representation at layer `l` be predicted from the current token embedding alone, without access to prior context?

This starter keeps the proposal's core workflow:

1. Extract token embeddings and hidden states from a pretrained causal language model.
2. Train a token-only linear probe for each layer.
3. Evaluate how prediction quality changes across layers.
4. Leave a clean extension point for later SAE-feature analysis.

## Group

- Melody Goldanloo

## Where Everything Lives

- `proposal.tex`: original proposal source.
- `proposal.pdf`: compiled proposal.
- `writeup/progress_report.md`: working writeup starter seeded from the proposal and shaped toward the course update.
- `writeup/references.bib`: starter bibliography file.
- `src/token_vs_context_llms/`: project code.
- `tests/`: lightweight unit tests for metrics and probe fitting.
- `configs/`: example experiment configs.
- `data/`: intended location for local datasets or manifests.
- `artifacts/`: extracted activations and metadata.
- `results/`: probe metrics, plots, and experiment summaries.

## Codebase Map

- `src/token_vs_context_llms/config.py`: experiment config dataclasses and YAML loader.
- `src/token_vs_context_llms/extract.py`: optional Hugging Face extraction pipeline for token embeddings and hidden states.
- `src/token_vs_context_llms/io.py`: artifact save/load helpers.
- `src/token_vs_context_llms/metrics.py`: cosine similarity, MSE, and `R^2`.
- `src/token_vs_context_llms/probe.py`: ridge-style linear probe fitting and per-layer evaluation.
- `src/token_vs_context_llms/cli.py`: command-line entrypoint exposed as `token-vs-context`.

## Quickstart

Use `uv` to create the environment and run the project.

```bash
uv sync --dev
uv run pytest
```

To enable the extraction pipeline as well:

```bash
uv sync --all-extras --dev
```

Example workflow:

```bash
uv run token-vs-context extract --config configs/small_debug.yaml
uv run token-vs-context probe --config configs/small_debug.yaml
```

The first command writes an `.npz` activation artifact under `artifacts/`. The second trains one token-only probe per layer and saves evaluation metrics under `results/`.

If `uv run` crashes on your local machine, the environment created by `uv sync` still works. You can temporarily use the installed binaries directly:

```bash
./.venv/bin/pytest
./.venv/bin/token-vs-context probe --config configs/small_debug.yaml
```

## Recommended Next Steps

- Replace the debug config with the actual model and dataset you want to study.
- Add an SAE extraction or loading path once you pick the specific SAE release.
- Expand `writeup/progress_report.md` with current experiments, roadblocks, and contributions.
- Add plotting code once you begin collecting real metrics across layers.

## Notes

- The extraction code is intentionally minimal and uses optional dependencies so the probe utilities and tests stay lightweight.
- If you switch to Gemma or another gated model, you may need Hugging Face authentication locally before running extraction.
