# Artifacts

Store extracted token-level activations here.

Expected files:

- `.npz` hidden-state artifacts produced by `token-vs-context extract`
- adjacent `.json` metadata files describing the extraction configuration

Current artifact contents:

- token embeddings
- per-layer hidden states
- input ids
- token positions
- token strings
- selected layer indices
- run metadata in the sidecar JSON

Future extension:

- add a parallel artifact path or format for SAE feature activations once the SAE stage begins
