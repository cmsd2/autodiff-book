PY := .venv/bin/python
export QUARTO_PYTHON := $(abspath $(PY))

.PHONY: all html pdf lean clean

all: lean html pdf

# Rendering runs every chapter's hidden checks; a failing check stops it.
html:
	quarto render --to html

pdf:
	quarto render --to pdf

# Machine-checked proofs of selected theorems.
lean:
	cd lean && lake build

clean:
	rm -rf _book .quarto
