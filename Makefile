.PHONY: test phase1-gate phase2-gate phase3-gate

PYTHON := .venv/bin/python

test:
	$(PYTHON) -m pytest -q

phase1-gate:
	$(PYTHON) -m eval.phase1_gate

phase2-gate:
	$(PYTHON) -m eval.phase2_gate

phase3-gate:
	$(PYTHON) -m eval.phase3_gate
