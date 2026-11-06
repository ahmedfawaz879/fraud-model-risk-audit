.DEFAULT_GOAL := help
PYTHON ?= python
DATA   ?= data/creditcard.csv
OUT    ?= audit_outputs
CONFIG ?= configs/credit_fraud.yaml

.PHONY: help install dev test lint format audit report clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and its runtime dependencies
	$(PYTHON) -m pip install -e .

dev:  ## Install with the development extras (pytest, ruff)
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite (synthetic data only; no dataset needed)
	$(PYTHON) -m pytest

lint:  ## Check style and imports
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:  ## Apply formatting
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

audit:  ## Run the full audit and write report, tables and figures to $(OUT)
	$(PYTHON) -m fraud_audit --config $(CONFIG) --data $(DATA) --output $(OUT)

report: audit  ## Run the audit and echo the generated report
	@cat $(OUT)/model_risk_audit_report.md

clean:  ## Remove generated artefacts and caches
	rm -rf $(OUT) .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
