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
| Hessian; its entries in two variables | $\nabla^2 f(\mathbf{a})$; $f_{xx}$, $f_{xy}$, $f_{yy}$ (never capitals, which are matrices) |
| eigenvalues of a symmetric matrix | $\lambda_1 \ge \dots \ge \lambda_n$, orthonormal eigenvectors $\mathbf{u}_1,\dots,\mathbf{u}_n$ |
| parameters, loss | $\boldsymbol{\theta}\in\mathbb{R}^p$, $L(\boldsymbol{\theta})$; in chapters 5 and 6 the function minimised is $f$, because $L$ is the smoothness constant there |
| loss of one example or one prediction | $\ell$ (a function of the output, or $\ell_t$ at position $t$) |
| layers of a network | $l = 1,\dots,D$, widths $n_l$ (not $\ell$, which is a loss) |
| learning rate, step | $\eta$, $k = 0, 1, 2, \dots$ |
| probability, expectation, variance | $P$, $\mathbb{E}$, $\operatorname{Var}$, written $\mathbb{E}[X]$ and $\operatorname{Var}[X]$ with square brackets |
| normal distribution | $N(\mu, s^2)$ with density $\varphi$ (`def-nn-density`); a weight's standard deviation in chapter 11 is $\sigma_w$ |
| likelihood, log-likelihood | $\mathcal{L}(\boldsymbol{\theta})$, $\ln\mathcal{L}(\boldsymbol{\theta})$ |
| vocabulary, its size, boundary token | $V$, $|V|$, $\diamond$; letters in typewriter type, $\mathtt{a}$ |
| sequence positions | $t = 1,\dots,T$; tokens $x_t$, with $x_1 = \diamond$; the prediction at position $t$ is of $x_{t+1}$ |
| attention | key positions $j$ (so weights $a_{tj}$, written $\alpha_{tj}$ in chapter 11), heads $i = 1,\dots,h$; in chapter 10 $V$ is also the matrix of values, which the chapter states |
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
that chapters can refer to each other. The table lists every
theorem-like and equation label that another chapter (or another
chapter's solutions) cites, together with the labels that were
promised when the chapters were planned. Keep them; if one must be
renamed, update every reference and this table. Add more labels
freely. Labels are global across the book, so give section and
equation labels a prefix naming the chapter's topic (for example
`sec-chain-...`, `eq-gd-...`), and before adding a generic label such
as `def-finite-probability` check that no other chapter defines it.
Every chapter's title label (`sec-several-variables`,
`sec-chain-rule`, ...) and its section labels may also be cited.

A concept is defined once, in the earliest chapter that needs it.
Later chapters recall it in a sentence and cite the label below; they
do not repeat the definition in italics or in a new block. A recap
must agree with the definition it recalls.

| Chapter | Labels |
|---|---|
| 1 | `thm-cauchy-schwarz` (Cauchy–Schwarz), `lem-linear-bound` ($\|A\mathbf{x}\| \le \|A\|_F\|\mathbf{x}\|$), `def-differentiable`, `eq-derivative` (the limit defining the derivative), `thm-differentiable-continuous`, `thm-jacobian` (the derivative is the Jacobian matrix of partial derivatives), `thm-continuous-partials` (continuous partial derivatives imply differentiability), `def-gradient`, `thm-directional` (directional derivative is $\nabla f\cdot\mathbf{u}$), `thm-steepest` (steepest ascent along the gradient), `exm-partials` ($x^2y + \sin(xy)$), `exm-steepest` (the bowl $x^2 + 3y^2$ at $(1, 1)$), `exm-affine`, `exm-elementwise`, `exm-quadratic` (derivatives of affine maps, componentwise functions and quadratic forms) |
| 2 | `lem-components` (differentiate component by component), `thm-chain-rule` ($J_{g\circ f} = J_g J_f$), `def-intermediate-variables`, `thm-chain-rule-paths` (derivative as a sum over paths), `eq-paths-step` (one step of the path recursion), `thm-product-rule`, `def-hessian`, `thm-mixed-partials` (the Hessian of a $C^2$ function is symmetric), `thm-taylor-second`, `eq-taylor-lagrange` and `eq-taylor-peano` (its two remainder forms), `def-local-extremum` (local extrema and critical points), `thm-critical-point` (gradient zero at a local extremum), `thm-spectral` (orthonormal eigenvectors of a symmetric matrix), `cor-quadratic-bounds` ($\lambda_n\|\mathbf{h}\|^2 \le \mathbf{h}^\top A\mathbf{h} \le \lambda_1\|\mathbf{h}\|^2$), `def-positive-definite` (positive and negative definite and semidefinite), `thm-second-derivative-test` (via Hessian eigenvalues) |
| 3 | `def-evaluation-trace`, `def-computational-graph`, `thm-topological-order`, `lem-trace-open` (the trace's domain is open), `eq-fwd-tangent` (the tangent recursion), `thm-forward-mode` (forward mode computes $J\mathbf{v}$), `exm-fwd-pass`, `def-dual-numbers`, `thm-dual-numbers` ($f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$) |
| 4 | `def-adjoint`, `thm-reverse-mode` (reverse accumulation computes the gradient on any acyclic graph), `thm-cheap-gradient` (gradient costs a constant multiple of evaluating $f$) |
| 5 | `def-l-smooth`, `lem-descent`, `thm-gd-nonconvex`, `thm-gd-convex`, `thm-gd-strongly-convex`, `eq-gd-quadratic` (the quadratic $\frac12\boldsymbol{\theta}^\top A\boldsymbol{\theta} - \mathbf{b}^\top\boldsymbol{\theta}$), `thm-gd-quadratic` (rates via eigenvalues and condition number), `exm-gd-zigzag` |
| 6 | `def-finite-probability` (finite probability space, random variables and vectors, expectation), `thm-expectation-linear` (linearity, and the formula over values), `def-independent-finite` (independence of two or several random vectors, pairwise independence), `lem-independent-functions` (functions of independent random vectors are independent), `thm-independent-product` ($\mathbb{E}[\mathbf{X}\cdot\mathbf{Y}] = \mathbb{E}[\mathbf{X}]\cdot\mathbb{E}[\mathbf{Y}]$ for independent vectors), `def-variance-finite` (variance and standard deviation), `lem-variance-formula` ($\operatorname{Var}[\mathbf{X}] = \mathbb{E}\|\mathbf{X}\|^2 - \|\mathbb{E}\mathbf{X}\|^2$, $\operatorname{Var}[c\mathbf{X} + \mathbf{v}] = c^2\operatorname{Var}[\mathbf{X}]$), `thm-variance-sum` (variances of pairwise independent vectors add), `def-sgd`, `thm-sgd-convergence`, `def-momentum`, `def-ema`, `thm-ema-bias` (bias correction), `def-adam` |
| 7 | `def-nn-layer`, `def-relu`, `def-mlp`, `thm-backprop-matrix` (backpropagation for an MLP in matrix form), `thm-universal-1d` (ReLU networks approximate continuous functions on an interval), `def-nn-density` (densities, the normal distribution $N(\mu, s^2)$, independence of real random variables), `thm-nn-init-variance` (variance of $W\mathbf{x}$ for random $W$) |
| 8 | `def-likelihood`, `eq-prob-nll` (the loss is the average negative log-likelihood), `def-entropy`, `def-cross-entropy`, `def-kl`, `thm-gibbs` (KL is non-negative), `thm-prob-categorical-mle` (maximum likelihood is relative frequency), `def-softmax`, `thm-softmax-jacobian`, `thm-softmax-ce-gradient` (gradient is $\mathbf{p} - \mathbf{y}$), `exm-uniform-loss` (uniform model has loss $\ln|V|$; the book's model starts near $3.35$) |
| 9 | `def-vocabulary` (tokens, boundary token $\diamond$), `thm-sequence-chain-rule` (chain rule of probability), `def-language-model` (numbering from $x_1 = \diamond$), `thm-bigram-mle` (maximum likelihood is relative frequency, row by row), `def-perplexity` (loss and perplexity of a text), `def-embedding` (one-hot vector and embedding) |
| 10 | `def-attention`, `eq-attention-matrix` ($\operatorname{Attn}(Q, K, V)$), `thm-attention-scale` (variance of $\mathbf{q}\cdot\mathbf{k}$ and the $1/\sqrt{d_h}$ scale), `def-causal-mask`, `thm-causal-cache` (causal attention at $t$ is attention over positions $1$ to $t$), `def-multi-head`, `thm-permutation-equivariance`, `def-position-embedding` (and the context length) |
| 11 | `def-rmsnorm`, `thm-residual-jacobian`, `def-small-gpt` (the model), `thm-parameter-count` (the model has 4192 parameters), `thm-initial-loss` (bounds on the expected loss at initialisation), `thm-temperature` (limits as temperature tends to 0 and to infinity), `thm-kv-cache` |
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
