"""
Lightning AI / local entry point — runs full pipeline in one command.
"""

import os
import sys
import urllib.request

# Fix paths for Lightning AI
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(ROOT, "crop_yield_prediction")
sys.path.insert(0, ROOT)
sys.path.insert(0, PROJ)

RAW_DIR = os.path.join(PROJ, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DATASETS = {
    "crop_recommendation.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/crop_recommendation.csv",
    "daily_temp.csv":          "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "monthly_sunspots.csv":    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
}

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

print("\n[Step 1] Generating processed data...")
from src.data_preprocessing import generate_and_save
generate_and_save()

print("\n[Step 2] Training all models...")
from src.train import main as train_main
train_main()

print("\n[Step 3] Evaluating...")
from src.evaluate import main as evaluate_main
evaluate_main()

print("\nDone. Results in crop_yield_prediction/outputs/results/")
