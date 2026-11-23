"""A NumPy version of the small transformer of chapter 11, for hidden checks.

The model follows the equations of the chapter: one layer, d = 16,
four heads of width 4, RMS normalisation with eps = 1e-5 and no gain,
a 16 -> 64 -> 16 ReLU MLP, no biases, and no normalisation before the
output matrix. ``forward_cached`` processes one position at a time and
keeps the keys and values of earlier positions in lists, which is the
order of computation the chapter's caching theorem compares with the
whole-sequence form shown in the chapter.
"""

import numpy as np

VOCAB = 27          # 26 letters and the boundary token
BOUNDARY = 26       # the boundary token comes after the letters
CONTEXT = 16
D = 16
HEADS = 4
HEAD_DIM = D // HEADS
HIDDEN = 4 * D
EPS = 1e-5
INIT_STD = 0.08

# (rows, columns) of each weight matrix; a matrix W maps x to W @ x.
SHAPES = {
    "E": (VOCAB, D),        # token embedding, one row per token
    "P": (CONTEXT, D),      # position embedding, one row per position
    "WQ": (D, D),
    "WK": (D, D),
    "WV": (D, D),
    "WO": (D, D),
    "W1": (HIDDEN, D),
    "W2": (D, HIDDEN),
    "WU": (VOCAB, D),       # output matrix, one row per token
}


def init_params(rng, std=INIT_STD):
    """Draw every weight independently from N(0, std^2)."""
    return {name: rng.normal(0.0, std, size=shape) for name, shape in SHAPES.items()}


def rms(x, eps=EPS):
    """RMS normalisation without a gain: x / sqrt(mean(x^2) + eps)."""
    return x / np.sqrt(np.mean(x * x) + eps)


def softmax(z):
    w = np.exp(z - np.max(z))
    return w / w.sum()


def encode(name):
    """Token sequence of a lower-case name, with the boundary token at both ends."""
    return [BOUNDARY] + [ord(c) - ord("a") for c in name] + [BOUNDARY]


def position_step(params, token, t, cache):
    """Logits at position t (counted from 0), appending this position's key and value."""
    r0 = rms(params["E"][token] + params["P"][t])
    u = rms(r0)
    q, k, v = params["WQ"] @ u, params["WK"] @ u, params["WV"] @ u
    cache.append((k, v))
    keys = np.array([kv[0] for kv in cache])
    values = np.array([kv[1] for kv in cache])
    out = np.empty(D)
    for j in range(HEADS):
        s = slice(j * HEAD_DIM, (j + 1) * HEAD_DIM)
        alpha = softmax(keys[:, s] @ q[s] / np.sqrt(HEAD_DIM))
        out[s] = alpha @ values[:, s]
    r1 = r0 + params["WO"] @ out
    r2 = r1 + params["W2"] @ np.maximum(params["W1"] @ rms(r1), 0.0)
    return params["WU"] @ r2, r2


def forward_cached(params, tokens):
    """Logits for every position, computed one position at a time with a cache."""
    cache = []
    return np.array([position_step(params, x, t, cache)[0] for t, x in enumerate(tokens)])


def residual_streams(params, tokens):
    """Final residual-stream vectors r^(2)_t for every position."""
    cache = []
    return np.array([position_step(params, x, t, cache)[1] for t, x in enumerate(tokens)])


def document_loss(params, name):
    """Average cross-entropy over the positions of one document."""
    tokens = encode(name)
    n = min(CONTEXT, len(tokens) - 1)
    logits = forward_cached(params, tokens[:n])
    losses = []
    for t in range(n):
        z = logits[t]
        m = z.max()
        losses.append(m + np.log(np.exp(z - m).sum()) - z[tokens[t + 1]])
    return float(np.mean(losses))
