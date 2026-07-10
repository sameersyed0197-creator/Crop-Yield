import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR


def set_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R², MAPE."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def format_results_table(results_dict):
    """Pretty-print model comparison table."""
    header = f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8}"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<22} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} "
              f"{metrics['R2']:>8.4f} {metrics['MAPE']:>8.2f}")
    print("=" * 60 + "\n")


def plot_confusion_matrix_regression(y_true, y_pred, save_path=None):
    """Residual plot for regression evaluation."""
    residuals = np.array(y_pred).flatten() - np.array(y_true).flatten()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="steelblue", s=20)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Yield (t/ha)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot — Fusion Model")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_model_summary(model, path):
    """Save Keras model summary to text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def create_output_dirs():
    """Create all output directories."""
    for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def log_experiment(config_dict, results_dict, filename="experiment_log.json"):
    """Save experiment config and results as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = {"config": config_dict, "results": results_dict}
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Experiment log saved to {path}")
