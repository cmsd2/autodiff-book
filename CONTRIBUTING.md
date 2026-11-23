# Writing guide

How the book is written, so that chapters by different hands read as
one book. Chapter 1 (`chapters/01-several-variables.qmd`) is the model
for everything here; read it before writing.

## Readers

The reader has finished a first university course in calculus,
vectors and matrices. Do not name any particular course, module or
institution in the book. They know single-variable calculus (including the mean value theorem
and Taylor polynomials), 2D and 3D vectors, matrices and their
products, eigenvalues and eigenvectors of small matrices, linear
transformations of the plane, first-order differential equations,
first- and second-order recurrence relations, counting and elementary
probability (equally likely outcomes), and direct, contradiction and
induction proofs. Anything beyond that must be developed in the text
or quoted with a precise reference.

## Shape of a chapter

- An opening of one to three paragraphs saying what the chapter does
  and why the book needs it. No signposting ("In this chapter we
  will...").
- Numbered sections (`## Title {#sec-...}`), each mixing narrative,
  definitions, theorems with proofs, and worked examples.
- A "Notes and further reading" section citing the bibliography.
- An "Exercises" section of 6 to 10 exercises, ranging from routine
  computation to short proofs. Every exercise has a worked solution in
  `chapters/solutions/_chNN.qmd`.

Use Quarto's cross-referenceable blocks: `::: {#def-...}`,
`::: {#thm-...}`, `::: {#lem-...}`, `::: {#cor-...}`, `::: {#exm-...}`,
`::: {#exr-...}`, with a `## Name` line inside when the result has a
name, and `::: {.proof}` for proofs. Number displayed equations that
are referred to: `$$ ... $$ {#eq-...}`.

## Introducing new material

Every new idea is led into before it is formalised. Section 1.1 of
chapter 1 ("Changing several inputs at once") is the model.

- Order: motivation (what question the idea answers), then a small
  concrete example with numbers that shows the idea at work, then the
  definition, then theorems and proofs, then further examples.
- Define every term before using it: in a `def-` block when the book
  uses the object or property again, otherwise in italics where it
  first appears. Introduce each symbol where it is defined, and never
  use one before that.
- Use standard mathematical terminology: increment, perturbation,
  partial derivative, Jacobian, adjoint, expectation, and so on. Avoid
  informal coinages such as "nudge". If an everyday word helps the
  intuition, pair it with the standard term the first time.
- When a definition is subtle, say why it is the right one by showing
  what goes wrong with the obvious alternative, as chapter 1 does with
  partial derivatives and differentiability.
- After displaying a formula, say in words what it means. For a matrix,
  say what its rows, columns and entries represent.
- Theorem statements stand on their own: restate the hypotheses rather
  than writing "under the assumptions above".

## Mathematics

- Prove what can be proved at this level, in full. When a proof is out
  of reach, state the result precisely, say that the book does not
  prove it, and cite where a proof can be found (book, chapter or
  theorem number).
- Theorem statements must be exactly true as stated: include every
  hypothesis (differentiability, convexity, positivity, non-zero
  denominators).
- Prefer a worked example right after each definition or theorem.
- Connect to the reader's background where it is natural: eigenvalues,
  recurrence relations, differential equations, counting.

## Notation

| Object | Notation |
|---|---|
| vectors | bold lower case, columns: $\mathbf{x} = (x_1,\dots,x_n)^\top$ |
| matrices | upper case italic: $A$, $W$; entries $a_{ij}$ |
| standard basis | $\mathbf{e}_1,\dots,\mathbf{e}_n$ |
| norm, dot product | $\|\mathbf{x}\|$ (Euclidean), $\mathbf{x}\cdot\mathbf{y}$ |
| derivative, Jacobian, gradient | $Df(\mathbf{a})$, $J_f(\mathbf{a})$, $\nabla f(\mathbf{a})$ |
| Hessian | $\nabla^2 f(\mathbf{a})$ |
| parameters, loss | $\boldsymbol{\theta}\in\mathbb{R}^p$, $L(\boldsymbol{\theta})$ |
| learning rate, step | $\eta$, $k = 0, 1, 2, \dots$ |
| probability, expectation, variance | $P$, $\mathbb{E}$, $\operatorname{Var}$ |
| vocabulary, its size | $V$, $|V|$ |
| sequence positions | $t = 1,\dots,T$; tokens $x_t$ |
| embedding width, heads, head width | $d$, $h$, $d_h = d/h$ |
| softmax | $\operatorname{softmax}(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$ |
| adjoint in reverse mode | $\bar v = \partial L/\partial v$ |

Use British spelling (normalise, optimisation).

## Prose style

- Active voice and direct verbs.
- No generic transitions ("Furthermore", "Moreover", "In conclusion").
- No signposting, no hedging, no filler ("in order to", "due to the
  fact that").
- No buzzwords, no flowery adjectives, no inflated significance.
- Avoid lists of exactly three adjectives, verbs or nouns in a row.
- Avoid em dashes; use commas, colons, parentheses or full stops. En
  dashes in names (Cauchy–Schwarz) and ranges are fine.
- Avoid the words "sharp", "grounded" and "load-bearing", and the
  pattern "not just X, it's Y".
- Show how parts fit together instead of calling them complex.

## Checks

Every computed example and every exercise solution gets a hidden check
immediately after it:

````markdown
```{python}
#| include: false
check(...)
```
````

The helpers in `checks/__init__.py` are `check`, `sym_equal`, `close`,
`maxima`, `maxima_zero` and `num_grad`. Each chapter starts with one
hidden cell importing what it needs. Check the specific numbers and
formulas the text states, not just related facts. Use Maxima as a
second opinion for identities the text relies on. A failing check must
stop the render: never wrap checks in `try`.

Visible code is welcome where it teaches, for example a small
automatic differentiation engine. Visible code must run at render
time and be checked by hidden cells. Write it from scratch; do not
copy code from other projects.

## Lean

Put machine-checked versions of a chapter's key results in
`lean/AutodiffBook/ChNN.lean` (namespace `AutodiffBook.ChNN`), with a
doc comment on each theorem naming the label it checks, for example
``/-- `thm-chain-rule`: ... -/``. Only include proofs that build;
never commit `sorry`. Prefer statements that match the book's theorem
closely, using Mathlib where it already has the result.

## References

Cite with `[@key]` or `@key [ch. 5]`. Each chapter adds its entries to
its own file, `bib/chNN.bib` (or `bib/appendix-a.bib`). Every entry
needs a DOI or a stable URL, checked against the publisher or
Crossref (`https://api.crossref.org/works/<doi>`); papers from arXiv
cite their arXiv identifier. Do not cite sources you have not seen.
Entries already in `bib/main.bib` can be cited from any chapter.

## Labels other chapters may cite

Each chapter defines at least these labels, with these meanings, so
that chapters written in parallel can refer to each other. Add more
labels freely; only these are guaranteed to exist. Labels are global
across the book, so give section and equation labels a prefix naming
the chapter's topic (for example `sec-chain-...`, `eq-gd-...`), and
before adding a generic label such as `def-finite-probability` check
that no other chapter defines it. Chapter 6 owns
`def-finite-probability`.

| Chapter | Labels |
|---|---|
| 1 (written) | `thm-cauchy-schwarz`, `lem-linear-bound`, `def-differentiable`, `thm-jacobian`, `thm-continuous-partials`, `def-gradient`, `thm-directional`, `thm-steepest`, `exm-affine`, `exm-elementwise`, `exm-quadratic` |
| 2 | `thm-chain-rule` (chain rule for maps between $\mathbb{R}^n$ and $\mathbb{R}^m$, $J_{g\circ f} = J_g J_f$), `thm-chain-rule-paths` (derivative as a sum over paths in a composition), `def-hessian`, `thm-taylor-second` (second-order Taylor in several variables), `thm-second-derivative-test` (via Hessian eigenvalues) |
| 3 | `def-computational-graph`, `def-evaluation-trace`, `def-dual-numbers`, `thm-dual-numbers` ($f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$), `thm-forward-mode` (forward mode computes $J\mathbf{v}$) |
| 4 | `def-adjoint`, `thm-reverse-mode` (reverse accumulation computes the gradient on any acyclic graph), `thm-cheap-gradient` (gradient costs a constant multiple of evaluating $f$) |
| 5 | `def-l-smooth`, `lem-descent`, `thm-gd-nonconvex`, `thm-gd-convex`, `thm-gd-strongly-convex`, `thm-gd-quadratic` (rates via eigenvalues and condition number) |
| 6 | `def-sgd`, `thm-sgd-convergence`, `def-momentum`, `def-ema`, `thm-ema-bias` (bias correction), `def-adam` |
| 7 | `def-mlp`, `def-relu`, `thm-backprop-matrix` (backpropagation for an MLP in matrix form), `thm-universal-1d` (ReLU networks approximate continuous functions on an interval) |
| 8 | `def-likelihood`, `def-entropy`, `def-cross-entropy`, `def-kl`, `thm-gibbs` (KL is non-negative), `def-softmax`, `thm-softmax-jacobian`, `thm-softmax-ce-gradient` (gradient is $\mathbf{p} - \mathbf{y}$) |
| 9 | `thm-sequence-chain-rule` (chain rule of probability), `thm-bigram-mle` (maximum likelihood is relative frequency), `def-perplexity`, `def-embedding` |
| 10 | `def-attention`, `thm-attention-scale` (variance of $\mathbf{q}\cdot\mathbf{k}$ and the $1/\sqrt{d_h}$ scale), `def-causal-mask`, `def-multi-head`, `thm-permutation-equivariance` |
| 11 | `def-rmsnorm`, `thm-residual-jacobian`, `thm-parameter-count` (the model has 4192 parameters), `thm-temperature` (limits as temperature tends to 0 and to infinity) |
| A | `thm-fd-error` (truncation and rounding error of finite differences), `thm-softmax-shift` (softmax is unchanged by adding a constant) |

## The model the book builds

Part 5 assembles a small transformer with the shape of Andrej
Karpathy's microgpt (February 2026,
<https://karpathy.github.io/2026/02/12/microgpt/>): a character-level
vocabulary of 27 tokens (26 letters and a boundary token), context
length 16, embedding width $d = 16$, $h = 4$ heads of width 4, one
layer, RMS normalisation without a learned gain, ReLU in a 16→64→16
MLP, no biases, weights initialised from $N(0, 0.08^2)$, and Adam with
$\beta_1 = 0.85$, $\beta_2 = 0.99$, $\epsilon = 10^{-8}$ and a learning
rate decaying linearly from 0.01. It trains on a list of names. Check
any detail against the source before stating it; the gist is
`gh gist view 8627fe009c40f57531cb18360106ce95`.

## Building

```bash
make html    # renders and runs every check
make pdf
make lean
```

To render one chapter while writing it:
`QUARTO_PYTHON=$PWD/.venv/bin/python quarto render chapters/NN-name.qmd --to html`.
Solutions render through `chapters/b-solutions.qmd`.
