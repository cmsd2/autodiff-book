"""Helpers for the hidden checks of chapters 3 and 4.

The chapters define visible code in labelled cells: the dual numbers of
chapter 3 (``dual-...``), and the tape engine (``tape-...``) and the
depth-first search over linked nodes (``linked-...``) of chapter 4.
Chapter 4 and the solutions appendix render in separate kernels, so they
load that code from the chapter source instead of keeping a copy that
could drift.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = re.compile(r"^```\{python\}\n(.*?)^```", re.M | re.S)


def chapter_code(chapter, prefix):
    """Return the code of every cell in ``chapter`` whose label starts with ``prefix``."""
    text = (ROOT / "chapters" / chapter).read_text()
    cells = []
    for body in CELL.findall(text):
        m = re.search(r"^#\| label: (\S+)", body, re.M)
        if m and m.group(1).startswith(prefix):
            cells.append(body)
    if not cells:
        raise LookupError(f"no cells labelled {prefix}* in {chapter}")
    return "\n".join(cells)


def load_chapter_code(chapter, prefix):
    """Run the labelled cells of ``chapter`` and return the names they define."""
    namespace = {}
    exec(chapter_code(chapter, prefix), namespace)
    return namespace


def tree_leaves(expr):
    """Number of occurrences of the variable in a formula written out as a tree.

    A formula is ``"x"``, a number, or a tuple ``(op, arg, ...)``.
    """
    if expr == "x":
        return 1
    if isinstance(expr, tuple):
        return sum(tree_leaves(a) for a in expr[1:])
    return 0


def tree_diff(expr):
    """Differentiate a tree formula by the sum, product and constant rules, without simplifying."""
    if expr == "x":
        return 1
    if not isinstance(expr, tuple):
        return 0
    op, *args = expr
    if op == "+":
        return ("+", tree_diff(args[0]), tree_diff(args[1]))
    if op == "-":
        return ("-", tree_diff(args[0]), tree_diff(args[1]))
    if op == "*":
        u, w = args
        if not isinstance(u, tuple) and u != "x":  # constant multiple rule
            return ("*", u, tree_diff(w))
        return ("+", ("*", tree_diff(u), w), ("*", u, tree_diff(w)))
    raise ValueError(op)
