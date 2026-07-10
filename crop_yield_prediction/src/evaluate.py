import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, PLOT_DIR, RESULTS_DIR, RANDOM_SEED, SEQUENCE_LENGTH
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import (
    normalize_images, normalize_weather, compute_vegetation_indices,
    extract_temporal_stats, create_ml_features
)
from src.utils import compute_metrics, format_results_table, create_output_dirs

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})


def load_all_models():
    models = {}
    for name in ["standalone_cnn", "standalone_lstm", "fusion_model"]:
        path = os.path.join(MODEL_DIR, f"{name}.keras")
        if os.path.exists(path):
            models[name] = keras.models.load_model(path, safe_mode=False)
            print(f"Loaded {name}")
    for name in ["random_forest", "svr"]:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name}")
    return models


def get_test_data():
    """
    Load test split and apply the SAME scalers used during training.
    Returns raw (t/ha) y_te and yield_sc so predictions can be inverse-transformed.
    """
    images, weather, yields, crop_labels = load_data()
    splits = split_data(images, weather, yields, crop_labels)

    img_sc   = joblib.load(os.path.join(MODEL_DIR, "image_scaler.pkl"))
    wx_sc    = joblib.load(os.path.join(MODEL_DIR, "weather_scaler.pkl"))
    yield_sc = joblib.load(os.path.join(MODEL_DIR, "yield_scaler.pkl"))

    img_te, _ = normalize_images(splits["test"]["images"],  fit=False, scaler=img_sc)
    wx_te, _  = normalize_weather(splits["test"]["weather"], fit=False, scaler=wx_sc)

    ndvi, _   = compute_vegetation_indices(splits["test"]["images"])
    stats     = extract_temporal_stats(ndvi)
    ml_te     = create_ml_features(stats, wx_te)

    # y_te stays in original t/ha — metrics computed on this
    y_te_raw  = splits["test"]["yields"]

    # NDVI seasonal plot uses full dataset
    img_all, _        = normalize_images(images, fit=False, scaler=img_sc)
    ndvi_all, _       = compute_vegetation_indices(img_all)
    ndvi_spatial_mean = ndvi_all.mean(axis=(2, 3))

    return img_te, wx_te, ml_te, y_te_raw, ndvi_spatial_mean, splits["test"]["crop_labels"], yield_sc


def predict_all(models, img_te, wx_te, ml_te, yield_sc):
    """
    Get predictions from all models and inverse-transform to original t/ha scale.
    Models were trained on scaled yields so raw output is in scaled space.
    """
    preds = {}

    def inv(p):
        return yield_sc.inverse_transform(
            np.array(p).flatten().reshape(-1, 1)
        ).flatten()

    if "standalone_cnn" in models:
        preds["standalone_cnn"] = inv(
            models["standalone_cnn"].predict(img_te, verbose=0).flatten()
        )
    if "standalone_lstm" in models:
        preds["standalone_lstm"] = inv(
            models["standalone_lstm"].predict(wx_te, verbose=0).flatten()
        )
    if "fusion_model" in models:
        preds["fusion_model"] = inv(
            models["fusion_model"].predict([img_te, wx_te], verbose=0).flatten()
        )
    if "random_forest" in models:
        preds["random_forest"] = inv(models["random_forest"].predict(ml_te))
    if "svr" in models:
        preds["svr"] = inv(models["svr"].predict(ml_te))

    return preds


def plot_comparison_table(results, save_path):
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "RMSE", "MAE", "R²", "MAPE%"]
    df = df.round(4)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    plt.title("Model Performance Comparison", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_predicted_vs_actual(y_true, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue", label="Predictions")
    lims = [min(y_true.min(), y_pred.min()) - 0.2, max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "r--", label="Perfect fit")
    ax.set_xlabel("Actual Yield (t/ha)")
    ax.set_ylabel("Predicted Yield (t/ha)")
    ax.set_title("Fusion Model — Predicted vs Actual")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_training_curves(histories, save_path):
    deep_models = ["standalone_cnn", "standalone_lstm", "fusion_model"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, deep_models):
        if name not in histories:
            ax.set_visible(False)
            continue
        h = histories[name]
        ax.plot(h["loss"], label="Train Loss")
        ax.plot(h["val_loss"], label="Val Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
    plt.suptitle("Training Loss Curves", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ndvi_seasonal(ndvi_spatial_mean, save_path):
    mean_curve = ndvi_spatial_mean.mean(axis=0)
    std_curve  = ndvi_spatial_mean.std(axis=0)
    weeks = np.arange(1, len(mean_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weeks, mean_curve, color="green", linewidth=2, label="Mean NDVI")
    ax.fill_between(weeks, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.3, color="green", label="±1 std")
    ax.set_xlabel("Week of Growing Season")
    ax.set_ylabel("NDVI")
    ax.set_title("Seasonal NDVI Trend (All Fields)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_importance(models, save_path):
    if "random_forest" not in models:
        return
    rf = models["random_forest"]
    importances = rf.feature_importances_
    top_n = min(20, len(importances))
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(top_n), importances[idx], color="olivedrab")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"F{i}" for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — Top Feature Importances")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_yield_distribution(y_true, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_true, bins=30, alpha=0.6, color="steelblue", label="Actual")
    ax.hist(y_pred, bins=30, alpha=0.6, color="darkorange", label="Predicted")
    ax.set_xlabel("Yield (t/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Yield Distribution — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    create_output_dirs()

    print("\n[Step 1] Loading models...")
    models = load_all_models()

    print("\n[Step 2] Preparing test data...")
    img_te, wx_te, ml_te, y_te, ndvi_spatial_mean, _, yield_sc = get_test_data()
    print(f"  y_te range: min={y_te.min():.2f}  max={y_te.max():.2f}  mean={y_te.mean():.2f}")

    print("\n[Step 3] Generating predictions (inverse-transformed to t/ha)...")
    preds = predict_all(models, img_te, wx_te, ml_te, yield_sc)

    # Sanity check — print raw prediction ranges
    for name, p in preds.items():
        print(f"  {name}: pred min={p.min():.2f}  max={p.max():.2f}  mean={p.mean():.2f}")

    print("\n[Step 4] Computing metrics...")
    results = {name: compute_metrics(y_te, p) for name, p in preds.items()}
    format_results_table(results)

    df = pd.DataFrame(results).T
    df.index.name = "Model"
    csv_path = os.path.join(RESULTS_DIR, "all_model_results.csv")
    df.to_csv(csv_path)
    print(f"Results CSV saved to {csv_path}")

    print("\n[Step 5] Generating plots...")
    plot_comparison_table(results, os.path.join(PLOT_DIR, "comparison_table.png"))

    if "fusion_model" in preds:
        plot_predicted_vs_actual(y_te, preds["fusion_model"],
                                 os.path.join(PLOT_DIR, "predicted_vs_actual.png"))
        plot_yield_distribution(y_te, preds["fusion_model"],
                                os.path.join(PLOT_DIR, "yield_distribution.png"))

    hist_path = os.path.join(RESULTS_DIR, "training_histories.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            histories = json.load(f)
        plot_training_curves(histories, os.path.join(PLOT_DIR, "training_loss_curves.png"))

    plot_ndvi_seasonal(ndvi_spatial_mean, os.path.join(PLOT_DIR, "ndvi_seasonal_trend.png"))
    plot_feature_importance(models, os.path.join(PLOT_DIR, "feature_importance.png"))

    print("\nEvaluation complete. All plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
