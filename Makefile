# ========= Makefile =========
# Simplify workflow commands

# Python virtual environment name
VENV = .venv

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - install dependencies"
	@echo "  make data         - download raw data (SPY, VIX)"
	@echo "  make features     - generate processed features"
	@echo "  make train        - train regime models"
	@echo "  make clean        - remove temp/data artifacts"

# --- Setup ---
install:
	pip install -r requirements.txt

# --- Data ---
data:
	python src/data.py --download --tickers SPY VIX --out data/raw/

features:
	python src/features.py --in data/raw/ --out data/processed/features.csv

train:
	python src/hmm_regimes.py --in data/processed/features.csv --out models/hmm.pkl

# --- Maintenance ---
clean:
	rm -rf data/raw/* data/processed/* models/* __pycache__
