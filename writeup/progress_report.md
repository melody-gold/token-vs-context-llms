# Token vs Context in LLM Representations

## Abstract

This project studies how much of a transformer's intermediate representation is recoverable from the current token alone, without access to prior context. The working method is to extract token embeddings and layer activations from a pretrained causal language model, then train token-only linear probes that predict each layer's hidden state from the input embedding. Probe quality across layers provides a quantitative picture of where token-local information remains dominant and where context-dependent structure becomes necessary. The long-term goal is to extend this analysis to sparse autoencoder features so that context dependence can be interpreted at the feature level rather than only at the activation-vector level.

## Introduction

Transformer representations mix information from the current token with information gathered from previous tokens through attention. An open interpretability question is how much of a given layer's representation reflects the token identity itself versus context-dependent computation.

This project asks:

How much of an LLM hidden representation can be predicted from the current token embedding alone?

This question connects to existing interpretability work in at least two ways:

- tuned-lens style work studies what is already present in intermediate activations
- sparse autoencoder work studies whether those activations can be decomposed into interpretable features

The contribution here is to invert the usual perspective and measure how much of those representations can be reconstructed without contextual information.

## Proposed Method

The planned workflow is:

1. Choose a pretrained causal language model and a text dataset.
2. Collect token embeddings and per-layer hidden states for token positions in the dataset.
3. Train a linear probe for each layer that maps the token embedding directly to the hidden representation.
4. Evaluate the probes with cosine similarity, mean squared error, and `R^2`.
5. Extend the same framework to SAE features once a specific SAE release is selected.

Interpretation:

- high predictability suggests a representation is largely token-local
- low predictability suggests contextual information is essential

## Relation To Existing Literature

The current framing is closest to:

- Belrose et al. on the tuned lens, which studies what latent predictions are already available in intermediate states
- Bricken et al. on sparse autoencoders and monosemanticity, which motivates feature-level analysis
- Olsson et al. on induction heads and in-context behavior, which motivates the expectation that later computations depend more strongly on prior tokens

## Current Experiments And Results

Current status:

- the repository now contains a minimal extraction and probing pipeline
- a debug config is provided for small local runs
- formal experiments have not been completed yet

Record completed experiments here as you run them. A useful format is:

| Date | Model | Dataset | Layers | Key result | What changed next |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Remaining Experiments

- run the small debug pipeline end-to-end on a lightweight model to validate extraction and storage
- choose the main model and dataset for the actual project run
- measure layerwise probe quality at scale
- inspect qualitative examples from well-predicted versus poorly predicted tokens
- extend the pipeline to SAE activations and compare feature-level predictability

## Expected Conclusions

The final conclusion will depend on whether prediction quality decays gradually or sharply across depth, and whether specific interpretable features stay token-local even when overall hidden-state predictability drops.

Possible outcomes:

- if early layers are highly predictable and later layers are not, that supports the view that contextual mixing deepens over the network
- if later layers remain strongly predictable, then token identity may explain more of the representation than expected
- if predictability is already weak in early layers, contextual computation may begin sooner than the initial hypothesis suggests

## Roadblocks

Known risks and open issues:

- a linear probe may be too weak, which could confound token-locality with probe underfitting
- extraction cost may become significant for larger models or larger datasets
- SAE selection may determine what kind of feature-level interpretation is possible
- model and dataset choices may be constrained by hardware or gated access

## Contributions

Update this section as the group finalizes responsibilities.

- Melody Goldanloo: proposal, starter repo scaffold, initial experiment design
- Add teammate names and concrete responsibilities here

## Appendix Planning

Use the appendix later for extra plots, derivations, ablations, or implementation details that are useful but not central to the main narrative.

## References

See `writeup/references.bib`.
