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

hmm:
	python src/hmm_regimes.py --input data/processed/features.csv --out-model models/hmm_model.pkl --out-plot figs/regimes_timeline.png

supervised:
	python src/prepare_supervised.py --input data/processed/features_with_regimes.csv --out data/processed/supervised_t5.csv --horizon 5

train:
	python src/lstm_predictor.py --input data/processed/supervised_t5.csv --epochs 50

evaluate:
	python src/eval.py --input data/processed/supervised_t5.csv --model models/lstm_best.pt --out-fig figs/confusion_matrix.png

# --- Maintenance ---
clean:
	rm -rf data/raw/* data/processed/* models/* __pycache__
