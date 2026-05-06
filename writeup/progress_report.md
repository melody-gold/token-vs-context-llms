# Token vs Context in LLM Representations

## Author: Melody Goldanloo

## Abstract

This project studies how much of a transformer's intermediate representation is recoverable from the current token alone, without access to prior context. The current baseline is to extract token embeddings and layer activations from a pretrained causal language model, then train token-only linear probes that predict each layer's hidden state from the input embedding. Probe quality across layers provides a quantitative picture of where token-local information remains dominant and where context-dependent structure becomes necessary. The planned second stage is to extend the same framework to sparse autoencoder (SAE) features so that context dependence can be interpreted at the feature level rather than only at the activation-vector level.

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
5. Extend the same framework to SAE features once a specific SAE release is selected for the chosen model.

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

- the repository contains a working extraction pipeline for token embeddings and hidden states
- extracted artifacts store token embeddings, hidden states, token ids, positions, tokens, layer indices, and run metadata
- an unregularized affine-probe evaluation path is implemented and tested
- a debug config is available for small local runs before scaling up
- proposal-scale experiments have not been completed yet

Record completed experiments here as you run them. A useful format is:

| Date | Model | Dataset | Layers | Key result | What changed next |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Remaining Experiments

- run the small debug pipeline end to end and confirm the saved artifact and metrics look sensible
- choose the main pretrained model for the baseline hidden-state experiment
- choose the main dataset slice for the project run, ideally something closer to the proposal target than the smoke-test config
- run layerwise probes at a larger token budget and generate a plot of cosine similarity, MSE, and `R^2` by layer
- check whether results are stable under different train/test seeds and token budgets
- optionally compare against ridge penalties after the affine baseline is established
- inspect representative well-predicted and poorly predicted tokens to understand failure modes
- select a model-compatible SAE release and add a feature-activation extraction or loading path
- compare hidden-state predictability against feature-level predictability once the SAE path is in place

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
- the main model choice may be constrained by local memory, model access, or token throughput
- SAE selection may determine what kind of feature-level interpretation is possible
- model and dataset choices may be constrained by hardware or gated access

## Questions

- Which main model should anchor the final report: Gemma, a smaller open model, or both?
- Which dataset slice is large enough to be meaningful but still feasible to extract locally?
- Should the final report prioritize a deeper hidden-state analysis first, or reserve more time for the SAE extension?
- What plots or qualitative examples will best support the context-versus-token interpretation?

## Appendix Planning

Use the appendix later for extra plots, derivations, ablations, or implementation details that are useful but not central to the main narrative.

## References

See `writeup/proposal/sources.bib`.
