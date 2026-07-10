"""
Lightning AI entry point — runs full pipeline:
  1. Download raw CSVs
  2. Generate processed data
  3. Train all 5 models
  4. Evaluate and save results
"""

import os
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "crop_yield_prediction", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DATASETS = {
    "crop_recommendation.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/crop_recommendation.csv",
    "daily_temp.csv":          "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "monthly_sunspots.csv":    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
}

# Try downloading; skip if already present
for fname, url in DATASETS.items():
    dest = os.path.join(RAW_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already present, skipping download.")
        continue
    print(f"  Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  {fname} downloaded.")
    except Exception as e:
        print(f"  WARNING: Could not download {fname}: {e}")
        print(f"  Please upload {fname} manually to {RAW_DIR}")

# Run pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crop_yield_prediction"))

print("\n[Step 1] Generating processed data...")
from src.data_preprocessing import generate_and_save
generate_and_save()

print("\n[Step 2] Training all models...")
from src.train import train_all
train_all()

print("\n[Step 3] Evaluating...")
from src.evaluate import evaluate_all
evaluate_all()

print("\nDone. Results in crop_yield_prediction/outputs/results/")
