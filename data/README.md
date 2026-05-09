# Data

Use this directory for dataset notes, manifests, cached subsets, and provenance records.

For this project, keep enough information here to answer:

- which dataset and split each experiment used
- whether the run was a smoke test or a proposal-scale run
- what filtering or truncation choices were applied
- any license or access constraints that affect reproducibility

Suggested layout:

- `data/raw/`: manually downloaded or exported assets.
- `data/processed/`: filtered subsets or analysis-ready tables.
- `data/notes.md`: provenance, licenses, and filtering decisions.

## Dataset Choice

The main hidden-state baseline can use `NeelNanda/pile-10k` from Hugging Face.
It is a 10,000-row debug subset of The Pile with a `train` split, a `text`
column, and a `meta` field containing the original Pile subset name.

Current config:

- `configs/pythia_pile10k.yaml`

Provenance notes:

- source: <https://huggingface.co/datasets/NeelNanda/pile-10k>
- split used initially: `train[:1000]`
- text column: `text`
- license listed by Hugging Face: `bigscience-bloom-rail-1.0`
