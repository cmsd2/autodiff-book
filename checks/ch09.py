"""Shared data and helpers for the checks in chapter 9 (language models).

The corpus is the five names written out in the chapter, so nothing is
downloaded at render time.
"""

from fractions import Fraction

import numpy as np

BOUNDARY = "."
CORPUS = ["anna", "ava", "emma", "eva", "ada"]

# The example vocabulary: the boundary token first, then the letters of the
# corpus in alphabetical order.
VOCAB = [BOUNDARY] + sorted(set("".join(CORPUS)))
INDEX = {s: i for i, s in enumerate(VOCAB)}


def wrap(name):
    """The token sequence of a name, with the boundary token at both ends."""
    return [BOUNDARY] + list(name) + [BOUNDARY]


def pairs(name):
    """Consecutive (previous, next) pairs of a wrapped name."""
    w = wrap(name)
    return list(zip(w[:-1], w[1:]))


def bigram_counts(names, vocab=VOCAB):
    """Matrix of pair counts n[a, b] over the vocabulary."""
    index = {s: i for i, s in enumerate(vocab)}
    n = np.zeros((len(vocab), len(vocab)), dtype=int)
    for name in names:
        for a, b in pairs(name):
            n[index[a], index[b]] += 1
    return n


def mle(counts):
    """Exact relative frequencies, row by row (rows with no counts left as None)."""
    rows = []
    for row in counts:
        total = int(row.sum())
        rows.append([Fraction(int(c), total) for c in row] if total else None)
    return rows


def add_one(counts):
    """Exact add-one smoothed transition probabilities."""
    m = counts.shape[1]
    return [[Fraction(int(c) + 1, int(row.sum()) + m) for c in row] for row in counts]


def sequence_probability(name, p):
    """Probability of a name under a bigram model p given as nested lists of Fractions."""
    prob = Fraction(1)
    for a, b in pairs(name):
        prob *= p[INDEX[a]][INDEX[b]]
    return prob


def average_nll(names, p):
    """Average negative log-likelihood per prediction (natural log), and perplexity."""
    logs = [float(np.log(float(p[INDEX[a]][INDEX[b]]))) for name in names for a, b in pairs(name)]
    loss = -sum(logs) / len(logs)
    return loss, float(np.exp(loss)), len(logs)


def rounds_to(x, value, places, message):
    """Check that ``x`` rounds to the stated ``value`` at ``places`` decimal places."""
    if round(float(x), places) != value:
        raise AssertionError(f"{message}: {x} does not round to {value}")
