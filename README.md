# From the Chain Rule to the Transformer

A theory-first textbook on the mathematics behind automatic
differentiation and small transformer language models, for readers
who have finished a first university course in calculus and matrices
(for example the Open University's MST124 and MST125).

## Layout

- `_quarto.yml`, `index.qmd`, `chapters/*.qmd`: the book, as a Quarto
  book that renders to HTML and PDF.
- `references.bib`: the bibliography. Entries carry DOIs checked
  against Crossref.
- `checks/`: helpers for the hidden checks. Chapters call them from
  cells marked `include: false`, using SymPy, NumPy and Maxima.
- `lean/`: Lean 4 proofs of selected theorems, against Mathlib
  v4.36.0. Each theorem's doc comment names the label of the result
  it checks in the text.

## Building

```bash
make html   # renders to _book/, running every hidden check
make pdf    # the same, through LuaLaTeX
make lean   # builds the Lean proofs
```

Rendering stops with an error if any check fails, so a successful
build means every computed example and exercise solution agreed with
SymPy and, where used, Maxima.

Requirements: Quarto, a TeX distribution with LuaLaTeX, Maxima,
uv (for the Python environment in `.venv`) and elan (for Lean). The
first `make lean` downloads Mathlib's prebuilt cache with
`lake exe cache get`.

## Licence

Released under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0). See `LICENSE`.
