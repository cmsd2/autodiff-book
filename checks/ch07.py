"""Simulation of the book's final model at initialisation, for chapters 7 and 8 (see exm-init-microgpt and exm-uniform-loss).

The model has the shape described in CONTRIBUTING.md: 27 tokens, context
16, width 16, 4 heads, one layer, RMS normalisation without a gain, a
16 -> 64 -> 16 ReLU MLP, no biases, and every weight drawn from
N(0, 0.08^2). Only the forward pass is needed, to measure the loss and
the size of intermediate vectors before any training.
"""

import numpy as np

VOCAB, CONTEXT, WIDTH, HEADS, STD = 27, 16, 16, 4, 0.08


def init_params(rng):
    shape = {
        "wte": (VOCAB, WIDTH), "wpe": (CONTEXT, WIDTH), "lm_head": (VOCAB, WIDTH),
        "wq": (WIDTH, WIDTH), "wk": (WIDTH, WIDTH), "wv": (WIDTH, WIDTH), "wo": (WIDTH, WIDTH),
        "fc1": (4 * WIDTH, WIDTH), "fc2": (WIDTH, 4 * WIDTH),
    }
    return {k: rng.normal(0.0, STD, size=s) for k, s in shape.items()}


def rmsnorm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def forward(params, tokens):
    """Return the logits at every position and a record of intermediate vectors."""
    T = len(tokens)
    dh = WIDTH // HEADS
    x = rmsnorm(params["wte"][tokens] + params["wpe"][np.arange(T)])
    res = x
    h = rmsnorm(x)
    q, k, v = h @ params["wq"].T, h @ params["wk"].T, h @ params["wv"].T
    out = np.zeros_like(q)
    for head in range(HEADS):
        s = slice(head * dh, (head + 1) * dh)
        scores = q[:, s] @ k[:, s].T / np.sqrt(dh)
        scores = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf)
        w = np.exp(scores - scores.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        out[:, s] = w @ v[:, s]
    x = out @ params["wo"].T + res
    res = x
    h = rmsnorm(x)
    pre = h @ params["fc1"].T
    mlp = np.maximum(pre, 0.0) @ params["fc2"].T
    x = mlp + res
    logits = x @ params["lm_head"].T
    return logits, {"mlp_in": h, "pre": pre, "mlp_out": mlp, "stream": res}


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
