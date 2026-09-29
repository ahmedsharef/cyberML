# ─────────────────────────────────────────────────────────────────────────────
# Makefile for ML-Based Cyberattack Detection System
# Usage: make <target>
# On Windows, use: nmake /f Makefile  OR  python -m <module> directly
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install train dashboard test clean help

## Install dependencies
install:
	pip install -r requirements.txt

## Run full training pipeline (preprocess + train all 5 models + generate report)
train:
	python -m src.train_models

## Launch the Streamlit dashboard
dashboard:
	streamlit run dashboard/app.py

## Run unit tests
test:
	pytest tests/ -v

## Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Remove all generated artefacts (processed data, models, reports)
clean:
	python -c "import shutil, pathlib; \
	[shutil.rmtree(p, ignore_errors=True) for p in [ \
	  'data/processed', 'models', 'reports/figures' \
	]]; \
	pathlib.Path('reports/model_comparison_report.md').unlink(missing_ok=True); \
	print('Cleaned.')"

## Show available commands
help:
	@echo "Available targets:"
	@echo "  make install    - pip install requirements"
	@echo "  make train      - run the full training pipeline"
	@echo "  make dashboard  - launch the Streamlit dashboard"
	@echo "  make test       - run unit tests"
	@echo "  make test-cov   - run tests with coverage report"
	@echo "  make clean      - remove generated files"
