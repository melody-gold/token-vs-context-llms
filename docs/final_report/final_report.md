# Token vs Context in LLM Representations

Melody Goldanloo

## Introduction

Large language models represent each token using both its identity and the
sequence of tokens that came before it. At the input layer, a token is represented
by an embedding lookup that depends only on the token id. After the model applies
attention and feed-forward blocks, however, the hidden state at that same token
position can contain information from earlier positions in the context. This
project studies where that transition becomes visible: how much of a transformer's
intermediate representation can be predicted from the current token alone, and how
much appears to require prior context?

The central experiment is a token-only reconstruction test. For each observed
token position in a dataset, I collect the model's input embedding and selected
hidden states. I then fit a simple affine probe from the token embedding to each
layer's hidden state. If the probe predicts a layer well, then that layer's
representation is largely recoverable from token identity alone under this probe.
If prediction quality is poor, the missing information is evidence that the layer
has become more context dependent. The goal is not to prove exactly what the model
uses internally, but to measure how much of the observed representation remains
linearly predictable from a context-free input.

This framing gives a quantitative way to compare layers. Early transformer layers
may remain close to lexical or local-token information, while later layers are
expected to mix in more contextual structure. A layerwise curve of mean squared
error, cosine similarity, and `R^2` can therefore show whether token-only
predictability decays gradually, drops sharply at a particular depth, or remains
surprisingly high throughout the network.

The project also leaves room for a feature-level extension using sparse
autoencoders. Hidden-state vectors are useful for measuring overall
predictability, but they are hard to interpret directly. Sparse autoencoder
features may make it possible to ask a more specific version of the same
question: which learned features are token-local, and which features require
context? The current implementation focuses first on the hidden-state baseline,
with the sparse-feature analysis planned as a second stage once the baseline
pipeline is stable.

## Literature Review

This project builds on work in transformer interpretability, probing, and sparse
feature decomposition. The common thread across these areas is the attempt to
understand what information is present inside a model's intermediate
representations, and how that information changes across layers.

The transformer architecture introduced by Vaswani et al. [@vaswani2017attention]
is the starting point for this question. Self-attention allows each token position
to combine information from other positions, so the representation at a position
is not limited to the token's own embedding. In causal language models, this
mixing is constrained to the current and previous tokens, making the distinction
between token-local information and prior-context information especially natural.
The experiment in this report uses that distinction directly: the probe receives
only the current token embedding, while the target hidden state was produced by a
model that had access to the earlier context.

Mechanistic interpretability work on transformer circuits motivates looking for
structured computations inside transformer layers rather than treating hidden
states as opaque vectors. Elhage et al. [@elhage2021framework] describe
transformers in terms of residual streams, attention heads, and feed-forward
components that can be analyzed as interacting computational units. That
perspective supports the idea that layerwise representations may contain a mix of
simple token-level components and more contextual computations. Olsson et al.
[@olsson2022induction] study induction heads and in-context learning, showing how
specific attention patterns can support behavior that depends on earlier tokens.
This motivates the expectation that later representations should often become
harder to recover from token identity alone.

The project is also related to work that asks what predictions are available from
intermediate model states. The tuned lens of Belrose et al.
[@belrose2023tunedlens] learns transformations from intermediate residual streams
to the model's output distribution, revealing how model predictions develop
across depth. My experiment uses a different direction of prediction. Instead of
asking what output distribution is already present in a hidden state, I ask how
much of the hidden state can be reconstructed from the context-free token
embedding. Both approaches use learned linear maps to make layerwise information
measurable, but they emphasize different relationships: hidden state to output in
the tuned lens, and token embedding to hidden state here.

Linear probing provides the methodological basis for this measurement. Alain and
Bengio [@alain2016understanding] use linear classifier probes to study information
available in intermediate layers, and Hewitt and Manning [@hewitt2019structural]
show that linear probes can recover syntactic structure from word
representations. These works support probing as a practical tool for studying
representations, but they also point to an important limitation: probe performance
measures what is recoverable by the probe, not necessarily what the model itself
uses. In this project, high token-only probe performance means that the hidden
state is linearly predictable from the token embedding. Low performance is
consistent with context dependence, but it could also reflect a nonlinear
relationship or limitations of the affine probe. For that reason, results should
be interpreted as evidence about linear recoverability rather than a complete
causal explanation.

Sparse autoencoder work provides the motivation for the planned feature-level
extension. Bricken et al. [@bricken2023monosemanticity] use dictionary learning to
decompose language model activations into sparse features that can be more
interpretable than raw activation dimensions. If suitable sparse autoencoders are
available for the chosen model, the same token-only predictability test can be
applied to feature activations. This would shift the analysis from asking whether
an entire hidden-state vector is token-local to asking which individual features
are token-local. That extension would help connect quantitative predictability
curves to more interpretable claims about the types of information represented by
the model.

Together, these lines of work motivate the project's baseline experiment. Prior
work shows that transformer layers develop meaningful intermediate states, that
linear probes can expose recoverable information in those states, and that sparse
methods may decompose activations into interpretable features. This project uses
those ideas to isolate a specific question: how much of a representation can be
explained by the current token before any context is used?

## References

See `writeup/proposal/sources.bib`.
