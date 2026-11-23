"""Shared helpers for the checks in chapter 10 (attention).

Everything here is plain NumPy, written for clarity rather than speed.
Vectors are 1-D arrays; a sequence of T vectors is a T x n array whose
row t is the vector at position t.
"""

import numpy as np


def softmax(z):
    """Softmax of a 1-D array."""
    e = np.exp(z - z.max())
    return e / e.sum()


def softmax_jacobian(a):
    """Jacobian of softmax at a point where its output is ``a``."""
    return np.diag(a) - np.outer(a, a)


def attend(q, K, V):
    """Attention for one query: weights and output. K is T x dh, V is T x dv."""
    a = softmax(K @ q / np.sqrt(len(q)))
    return a, V.T @ a


def attention(Q, K, V, causal=False):
    """Matrix form: row t of the result is the attention output for query t."""
    T, dh = Q.shape
    S = Q @ K.T / np.sqrt(dh)
    if causal:
        S = np.where(np.tril(np.ones((T, K.shape[0]), dtype=bool)), S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(axis=1, keepdims=True)
    return A, A @ V


def multi_head(X, WQ, WK, WV, WO, h, causal=False):
    """Multi-head self-attention of the rows of X; returns the T x d output."""
    d = X.shape[1]
    dh = d // h
    Q, K, V = X @ WQ.T, X @ WK.T, X @ WV.T
    heads = []
    for i in range(h):
        cols = slice(i * dh, (i + 1) * dh)
        heads.append(attention(Q[:, cols], K[:, cols], V[:, cols], causal)[1])
    return np.hstack(heads) @ WO.T


def multi_head_cached(X, WQ, WK, WV, WO, h):
    """The same causal computation one position at a time with a key-value cache."""
    d = X.shape[1]
    dh = d // h
    keys, values, out = [], [], []
    for x in X:
        q, k, v = WQ @ x, WK @ x, WV @ x
        keys.append(k)
        values.append(v)
        Kc, Vc = np.array(keys), np.array(values)
        o = np.concatenate([attend(q[i * dh:(i + 1) * dh], Kc[:, i * dh:(i + 1) * dh],
                                   Vc[:, i * dh:(i + 1) * dh])[1] for i in range(h)])
        out.append(WO @ o)
    return np.array(out)


def rmsnorm(x, eps=1e-5):
    return x / np.sqrt((x * x).mean() + eps)


def small_gpt_logits(tokens, params, use_positions=True, h=4):
    """Logits at every position of a one-layer model with the shape of the book's model."""
    E, P, WQ, WK, WV, WO, W1, W2, Wout = (params[k] for k in
                                          ("E", "P", "WQ", "WK", "WV", "WO", "W1", "W2", "Wout"))
    X = np.array([E[x] + (P[t] if use_positions else 0) for t, x in enumerate(tokens)])
    X = np.array([rmsnorm(x) for x in X])
    Xn = np.array([rmsnorm(x) for x in X])
    X = X + multi_head(Xn, WQ, WK, WV, WO, h, causal=True)
    Xn = np.array([rmsnorm(x) for x in X])
    X = X + np.maximum(Xn @ W1.T, 0) @ W2.T
    return X @ Wout.T


def random_params(rng, V=27, d=16, T=16, std=0.08):
    shapes = {"E": (V, d), "P": (T, d), "WQ": (d, d), "WK": (d, d), "WV": (d, d), "WO": (d, d),
              "W1": (4 * d, d), "W2": (d, 4 * d), "Wout": (V, d)}
    return {k: rng.normal(scale=std, size=s) for k, s in shapes.items()}


def saturation_stats(dh, scaled, rng, T=8, trials=20000):
    """Score spread and softmax saturation for random queries and keys.

    Returns the standard deviation of the scores, the mean of the largest
    weight, the median of the largest absolute Jacobian entry, and the
    fraction of trials whose largest weight exceeds 0.99.
    """
    q = rng.normal(size=(trials, dh))
    k = rng.normal(size=(trials, T, dh))
    s = np.einsum("nd,ntd->nt", q, k)
    if scaled:
        s = s / np.sqrt(dh)
    s = s - s.max(axis=1, keepdims=True)
    a = np.exp(s)
    a = a / a.sum(axis=1, keepdims=True)
    J = np.einsum("ni,ij->nij", a, np.eye(T)) - np.einsum("ni,nj->nij", a, a)
    jmax = np.abs(J).max(axis=(1, 2))
    return (float(np.einsum("nd,ntd->nt", q, k).std() / (np.sqrt(dh) if scaled else 1)),
            float(a.max(axis=1).mean()), float(np.median(jmax)),
            float((a.max(axis=1) > 0.99).mean()))
