SHELL := /bin/bash

# Config
VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
STREAMLIT  := $(VENV)/bin/streamlit
PYTEST     := $(VENV)/bin/pytest
REQS       := requirements.txt
DEPS_STAMP := $(VENV)/.deps-installed

.PHONY: run dev install venv clean reset help test

help:
	@echo "make run     - ensure venv+deps, then run Streamlit"
	@echo "make install - ensure venv+deps only"
	@echo "make test    - run pytest"
	@echo "make clean   - remove venv"
	@echo "make reset   - clean + reinstall"
	@echo "make dev     - alias of run"

# 1) Create virtualenv if missing
$(VENV):
	python3 -m venv $(VENV)

# 2) Install/refresh deps when requirements.txt changes (or first time)
$(DEPS_STAMP): $(REQS) | $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQS)
	@touch $(DEPS_STAMP)

# Convenience alias
install: $(DEPS_STAMP)

# 3) Run the app
run: $(DEPS_STAMP)
	$(STREAMLIT) run app.py

dev: run

# 4) Tests
test: $(DEPS_STAMP)
	PYTHONPATH=. $(PYTEST) -q

# Housekeeping
clean:
	rm -rf $(VENV)

reset: clean install

