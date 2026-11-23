"""Helpers for the hidden checks in the book's chapters.

Chapters call these from cells marked ``include: false``, so a false
claim stops the render instead of reaching the reader. SymPy does most
of the symbolic work, and Maxima gives a second opinion on identities
that the text relies on.
"""

import subprocess

import numpy as np
import sympy as sp


def check(condition, message):
    """Fail the render with ``message`` unless ``condition`` holds."""
    if not condition:
        raise AssertionError(message)


def sym_equal(a, b, message):
    """Check that two SymPy expressions are identical after simplification."""
    check(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0, f"{message}: {a} != {b}")


def maxima(expr):
    """Evaluate ``expr`` in a fresh Maxima session and return the result as text."""
    script = f'display2d:false$ linel:100000$ print("@@", {expr})$'
    run = subprocess.run(
        ["maxima", "--very-quiet", "--batch-string", script],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in run.stdout.splitlines():
        if line.startswith("@@"):
            return line[2:].strip()
    raise RuntimeError(f"no result from Maxima for {expr!r}:\n{run.stdout}")


def maxima_zero(expr, message):
    """Check that Maxima simplifies ``expr`` to 0."""
    result = maxima(f"ratsimp(trigsimp({expr}))")
    check(result == "0", f"{message}: Maxima gives {result}")


def close(a, b, message, tol=1e-8):
    """Check that two numbers or arrays agree to a relative tolerance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = 1.0 + np.max(np.abs(b))
    check(np.max(np.abs(a - b)) <= tol * scale, f"{message}: {a} != {b}")


def num_grad(f, x, h=1e-6):
    """Central-difference estimate of the gradient of ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g
