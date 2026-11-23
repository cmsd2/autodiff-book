"""Simulation of the book's final model at initialisation, for chapters 7 and 8 (see exm-init-microgpt and exm-uniform-loss).

The model is the one of chapter 11 (def-small-gpt), run by checks.ch11:
27 tokens, context 16, width 16, 4 heads, one layer, RMS normalisation
without a gain, a 16 -> 64 -> 16 ReLU MLP, no biases, and every weight
drawn from N(0, 0.08^2). Only the forward pass is needed, to measure the
loss and the size of intermediate vectors before any training.
"""

import numpy as np

from checks.ch11 import VOCAB, init_params, run

__all__ = ["VOCAB", "init_params", "forward", "initial_losses"]


def forward(params, tokens):
    """Return the logits at every position and a record of intermediate vectors.

    The record holds, with one row per position, the normalised input of the
    MLP ("mlp_in"), its pre-activations ("pre"), its output ("mlp_out") and
    the residual stream that the MLP output is added to ("stream").
    """
    rec = run(params, tokens)
    return rec["z"], {"mlp_in": rec["h"], "pre": rec["pre"], "mlp_out": rec["mlp_out"],
                      "stream": rec["r1"]}


def initial_losses(rng, n_models=200, length=8):
    """Mean cross-entropy loss of freshly initialised models on random sequences."""
    losses, logit_sd = [], []
    for _ in range(n_models):
        params = init_params(rng)
        tokens = rng.integers(0, VOCAB, size=length + 1)
        logits, _ = forward(params, tokens[:-1])
        m = logits.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))[:, 0]
        losses.append(np.mean(lse - logits[np.arange(length), tokens[1:]]))
        logit_sd.append(logits.std())
    return np.array(losses), np.array(logit_sd)
