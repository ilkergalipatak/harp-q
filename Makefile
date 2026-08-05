PYTHON ?= python3

.PHONY: test smoke demo check

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m harpq.cli doctor --json

demo:
	PYTHONPATH=src $(PYTHON) -m harpq.cli demo --output artifacts/demo

check: test smoke
