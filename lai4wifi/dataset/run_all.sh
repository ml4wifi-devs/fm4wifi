#!/bin/bash

echo "Running all dataset generation scripts..."

echo "[1/4] Running random configurations generation..."
python random_tx.py

echo "[2/4] Running H-MAB configurations generation..."
python h_mab.py

echo "[3/4] Running F-Optimal configurations generation..."
python f_optimal.py

echo "[4/4] Running T-Optimal configurations generation..."
python t_optimal.py
