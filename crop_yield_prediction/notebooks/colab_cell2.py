# ============================================================
# CELL 2 — Train All 5 Models + Evaluate + Save Results
# (Self-contained — all config defined here)
# ============================================================

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, keras
from keras import layers, Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

# ── Config (copy of Cell 1 values) ───────────────────────────
PROCESSED_DIR        = "/content/processed"
MODEL_DIR            = "/content/models"
PLOT_DIR             = "/content/plots"
RESULTS_DIR          = "/content/results"
NUM_FIELDS           = 2000
SEQUENCE_LENGTH      = 16
NUM_BANDS            = 5
NUM_WEATHER_FEATURES = 6
IMAGE_HEIGHT         = 32
IMAGE_WIDTH          = 32
DROPOUT_RATE         = 0.2
LEARNING_RATE        = 0.001
BATCH_SIZE           = 32
EPOCHS               = 80
RANDOM_SEED          = 42
TEST_SPLIT           = 0.15
VALIDATION_SPLIT     = 0.20
EARLY_STOPPING_PATIENCE = 20

for d in [MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

keras.utils.set_random_seed(RANDOM_SEED)

# ── Load data ─────────────────────────────────────────────────
images      = np.load(f"{PROCESSED_DIR}/images.npy")
weather     = np.load(f"{PROCESSED_DIR}/weather.npy")
yields      = np.load(f"{PROCESSED_DIR}/yields.npy")
crop_labels = np.load(f"{PROCESSED_DIR}/crop_labels.npy")
ndvi_store  = np.load(f"{PROCESSED_DIR}/ndvi.npy")

print(f"images      : {images.shape}")
print(f"weather     : {weather.shape}")
print(f"yields      : {yields.shape}  mean={yields.mean():.2f}  std={yields.std():.2f}")

# ── Split ─────────────────────────────────────────────────────
rng2   = np.random.default_rng(RANDOM_SEED)
idx    = rng2.permutation(NUM_FIELDS)
n_test = int(NUM_FIELDS * TEST_SPLIT)
n_val  = int(NUM_FIELDS * VALIDATION_SPLIT)
te_idx = idx[:n_test]
va_idx = idx[n_test:n_test+n_val]
tr_idx = idx[n_test+n_val:]
print(f"Train:{len(tr_idx)}  Val:{len(va_idx)}  Test:{len(te_idx)}")

# ── Scalers ───────────────────────────────────────────────────
N, T, H, W, C = images.shape
_, _, F        = weather.shape

img_sc  = MinMaxScaler()
img_tr  = img_sc.fit_transform(images[tr_idx].reshape(-1,C)).reshape(len(tr_idx),T,H,W,C).astype(np.float32)
img_va  = img_sc.transform(images[va_idx].reshape(-1,C)).reshape(len(va_idx),T,H,W,C).astype(np.float32)
img_te  = img_sc.transform(images[te_idx].reshape(-1,C)).reshape(len(te_idx),T,H,W,C).astype(np.float32)
joblib.dump(img_sc, f"{MODEL_DIR}/image_scaler.pkl")

wx_sc   = StandardScaler()
wx_tr   = wx_sc.fit_transform(weather[tr_idx].reshape(-1,F)).reshape(len(tr_idx),T,F).astype(np.float32)
wx_va   = wx_sc.transform(weather[va_idx].reshape(-1,F)).reshape(len(va_idx),T,F).astype(np.float32)
wx_te   = wx_sc.transform(weather[te_idx].reshape(-1,F)).reshape(len(te_idx),T,F).astype(np.float32)
joblib.dump(wx_sc, f"{MODEL_DIR}/weather_scaler.pkl")

y_tr = yields[tr_idx]
y_va = yields[va_idx]
y_te = yields[te_idx]

# ── ML flat features (for RF + SVR) ──────────────────────────
ndvi_sp   = ndvi_store.mean(axis=(2,3))           # (N, T)
peak_n    = ndvi_sp.max(axis=1)
mean_n    = ndvi_sp.mean(axis=1)
t_ax      = np.arange(T, dtype=np.float32)
slope_n   = np.array([np.polyfit(t_ax, ndvi_sp[i],1)[0] for i in range(NUM_FIELDS)], dtype=np.float32)
gup_w     = ndvi_sp.argmax(axis=1).astype(np.float32)
half_p    = peak_n / 2
sen_w     = np.array([
    next((t for t in range(int(gup_w[i]),T) if ndvi_sp[i,t] < half_p[i]), T-1)
    for i in range(NUM_FIELDS)
], dtype=np.float32)
ndvi_stats = np.stack([peak_n, mean_n, slope_n, gup_w, sen_w], axis=1)
ml_feat    = np.concatenate([ndvi_stats, weather.reshape(NUM_FIELDS,-1)], axis=1).astype(np.float32)
ml_tr      = ml_feat[tr_idx]
ml_va      = ml_feat[va_idx]
ml_te      = ml_feat[te_idx]

# ── Model builders ────────────────────────────────────────────
def build_cnn_encoder():
    inp = Input(shape=(H, W, C))
    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(128, activation="relu")(x)
    return Model(inp, out, name="cnn_enc")

def build_standalone_cnn():
    inp = Input(shape=(T, H, W, C), name="image_input")
    x   = layers.TimeDistributed(build_cnn_encoder())(inp)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1,  activation="linear")(x)
    m   = Model(inp, out, name="standalone_cnn")
    m.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return m

def build_standalone_lstm():
    inp = Input(shape=(T, F), name="weather_input")
    x   = layers.Masking(mask_value=0.0)(inp)
    x   = layers.LSTM(128, return_sequences=True)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1,  activation="linear")(x)
    m   = Model(inp, out, name="standalone_lstm")
    m.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return m

def build_fusion():
    # --- CNN branch (spatial) ---
    img_inp = Input(shape=(T, H, W, C), name="image_input")
    cnn_enc = build_cnn_encoder()
    sp_seq  = layers.TimeDistributed(cnn_enc, name="td_cnn")(img_inp)
    sp_feat = layers.LSTM(64, name="spatial_lstm")(sp_seq)          # (64,)
    sp_proj = layers.Dense(64, activation="relu", name="cnn_proj")(sp_feat)

    # --- LSTM branch (weather) ---
    wx_inp  = Input(shape=(T, F), name="weather_input")
    x2      = layers.Masking(mask_value=0.0)(wx_inp)
    x2      = layers.LSTM(128, return_sequences=True)(x2)
    x2      = layers.Dropout(DROPOUT_RATE)(x2)
    x2      = layers.LSTM(64)(x2)
    x2      = layers.Dropout(DROPOUT_RATE)(x2)
    wx_feat = layers.Dense(64, activation="relu", name="lstm_proj")(x2)  # (64,)

    # --- Attention gate: learn per-branch weights ---
    concat_gate = layers.Concatenate(name="gate_input")([sp_proj, wx_feat])  # (128,)
    gate = layers.Dense(2, activation="softmax", name="branch_gate")(concat_gate)  # (2,)
    cnn_w  = layers.Lambda(lambda g: g[:, 0:1], name="cnn_weight")(gate)   # (1,)
    lstm_w = layers.Lambda(lambda g: g[:, 1:2], name="lstm_weight")(gate)  # (1,)
    cnn_scaled  = layers.Multiply(name="cnn_scaled")([sp_proj,  cnn_w])
    lstm_scaled = layers.Multiply(name="lstm_scaled")([wx_feat, lstm_w])

    fused = layers.Concatenate(name="fusion")([cnn_scaled, lstm_scaled])   # (128,)
    x = layers.Dense(128, activation="relu")(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)
    m = Model(inputs=[img_inp, wx_inp], outputs=out, name="fusion_model")
    m.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return m

def get_callbacks(name):
    return [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"{MODEL_DIR}/{name}_best.keras",
                        monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred)**2))),
        "MAE":  float(np.mean(np.abs(y_true - y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    }

histories = {}
results   = {}

# ── 1. Standalone CNN ─────────────────────────────────────────
print("\n" + "="*50 + "\nTraining standalone_cnn...\n" + "="*50)
m_cnn = build_standalone_cnn()
h = m_cnn.fit(img_tr, y_tr, validation_data=(img_va, y_va),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=get_callbacks("standalone_cnn"), verbose=1)
m_cnn.save(f"{MODEL_DIR}/standalone_cnn.keras")
histories["standalone_cnn"] = {k:[float(v) for v in vals] for k,vals in h.history.items()}
results["standalone_cnn"]   = compute_metrics(y_te, m_cnn.predict(img_te, verbose=0))
print(f"CNN   RMSE={results['standalone_cnn']['RMSE']:.4f}  R2={results['standalone_cnn']['R2']:.4f}")

# ── 2. Standalone LSTM ────────────────────────────────────────
print("\n" + "="*50 + "\nTraining standalone_lstm...\n" + "="*50)
m_lstm = build_standalone_lstm()
h = m_lstm.fit(wx_tr, y_tr, validation_data=(wx_va, y_va),
               epochs=EPOCHS, batch_size=BATCH_SIZE,
               callbacks=get_callbacks("standalone_lstm"), verbose=1)
m_lstm.save(f"{MODEL_DIR}/standalone_lstm.keras")
histories["standalone_lstm"] = {k:[float(v) for v in vals] for k,vals in h.history.items()}
results["standalone_lstm"]   = compute_metrics(y_te, m_lstm.predict(wx_te, verbose=0))
print(f"LSTM  RMSE={results['standalone_lstm']['RMSE']:.4f}  R2={results['standalone_lstm']['R2']:.4f}")

# ── 3. Fusion model (two-phase training) ─────────────────────
print("\n" + "="*50 + "\nTraining fusion_model...\n" + "="*50)
m_fus = build_fusion()

# Phase 1: freeze CNN branch, warm up LSTM + gate (20 epochs)
for layer in m_fus.layers:
    if layer.name in ("td_cnn", "spatial_lstm", "cnn_proj"):
        layer.trainable = False
m_fus.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
print("Phase 1: warming up LSTM branch (CNN frozen)...")
m_fus.fit([img_tr, wx_tr], y_tr,
          validation_data=([img_va, wx_va], y_va),
          epochs=20, batch_size=BATCH_SIZE, verbose=1)

# Phase 2: unfreeze all, fine-tune jointly
for layer in m_fus.layers:
    layer.trainable = True
m_fus.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE * 0.3), loss="mse", metrics=["mae"])
print("Phase 2: joint fine-tuning (all layers unfrozen)...")
h = m_fus.fit([img_tr, wx_tr], y_tr,
              validation_data=([img_va, wx_va], y_va),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=get_callbacks("fusion_model"), verbose=1)
m_fus.save(f"{MODEL_DIR}/fusion_model.keras")
histories["fusion_model"] = {k:[float(v) for v in vals] for k,vals in h.history.items()}
results["fusion_model"]   = compute_metrics(y_te, m_fus.predict([img_te, wx_te], verbose=0))
print(f"Fusion RMSE={results['fusion_model']['RMSE']:.4f}  R2={results['fusion_model']['R2']:.4f}")

# ── 4. Random Forest ──────────────────────────────────────────
print("\nTraining random_forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(ml_tr, y_tr)
joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
results["random_forest"] = compute_metrics(y_te, rf.predict(ml_te))
print(f"RF    RMSE={results['random_forest']['RMSE']:.4f}  R2={results['random_forest']['R2']:.4f}")

# ── 5. SVR ────────────────────────────────────────────────────
print("\nTraining svr...")
svr = SVR(kernel="rbf", C=10, epsilon=0.1)
svr.fit(ml_tr, y_tr)
joblib.dump(svr, f"{MODEL_DIR}/svr.pkl")
results["svr"] = compute_metrics(y_te, svr.predict(ml_te))
print(f"SVR   RMSE={results['svr']['RMSE']:.4f}  R2={results['svr']['R2']:.4f}")

# ── Save results ──────────────────────────────────────────────
with open(f"{RESULTS_DIR}/training_histories.json", "w") as f:
    json.dump(histories, f, indent=2)
df_res = pd.DataFrame(results).T
df_res.index.name = "Model"
df_res.to_csv(f"{RESULTS_DIR}/all_model_results.csv")
print(f"\nResults saved to {RESULTS_DIR}/all_model_results.csv")

# ── Final results table ───────────────────────────────────────
best_r2 = max(m["R2"] for m in results.values())
print("\n" + "="*65)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'MAPE%':>8}")
print("-"*65)
for name, m in results.items():
    tag = " <-- BEST" if m["R2"] == best_r2 else ""
    print(f"{name:<22} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f} {m['MAPE']:>8.2f}{tag}")
print("="*65)

# ── Plots ─────────────────────────────────────────────────────
y_pred_fus  = m_fus.predict([img_te, wx_te], verbose=0).flatten()
y_pred_lstm = m_lstm.predict(wx_te, verbose=0).flatten()
y_pred_rf   = rf.predict(ml_te)

# Plot 1: Predicted vs Actual
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(y_te, y_pred_fus, alpha=0.6, color="steelblue", s=25, label="Fusion")
lims = [min(y_te.min(), y_pred_fus.min())-0.2, max(y_te.max(), y_pred_fus.max())+0.2]
ax.plot(lims, lims, "r--", label="Perfect fit")
ax.set_xlabel("Actual Yield (t/ha)"); ax.set_ylabel("Predicted Yield (t/ha)")
ax.set_title("Fusion Model — Predicted vs Actual"); ax.legend()
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/predicted_vs_actual.png", dpi=300); plt.close()

# Plot 2: Training loss curves
fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, name in zip(axes, ["standalone_cnn","standalone_lstm","fusion_model"]):
    if histories.get(name):
        ax.plot(histories[name]["loss"],     label="Train")
        ax.plot(histories[name]["val_loss"], label="Val")
        ax.set_title(name); ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss"); ax.legend(fontsize=8)
plt.suptitle("Training Loss Curves"); plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/training_loss_curves.png", dpi=300); plt.close()

# Plot 3: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(12,5))
names  = list(results.keys())
rmses  = [results[n]["RMSE"] for n in names]
r2s    = [results[n]["R2"]   for n in names]
colors = ["#4a7c59","#8fbc8f","#2d8a4e","#c8a96e","#a0522d"]
axes[0].bar(names, rmses, color=colors)
axes[0].set_title("RMSE — lower is better")
axes[0].set_xticklabels(names, rotation=20, ha="right")
axes[1].bar(names, r2s, color=colors)
axes[1].set_title("R2 Score — higher is better")
axes[1].set_xticklabels(names, rotation=20, ha="right")
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/model_comparison.png", dpi=300); plt.close()

# Plot 4: NDVI seasonal trend
ndvi_sp_mean = ndvi_store.mean(axis=(2,3))
mc = ndvi_sp_mean.mean(axis=0); sc = ndvi_sp_mean.std(axis=0)
weeks = np.arange(1, SEQUENCE_LENGTH+1)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(weeks, mc, color="green", linewidth=2, label="Mean NDVI")
ax.fill_between(weeks, mc-sc, mc+sc, alpha=0.3, color="green", label="+/-1 std")
ax.set_xlabel("Week of Growing Season"); ax.set_ylabel("NDVI")
ax.set_title("Seasonal NDVI Trend (All Fields)"); ax.legend()
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/ndvi_seasonal_trend.png", dpi=300); plt.close()

# Plot 5: RF Feature Importance
importances = rf.feature_importances_
idx_top = np.argsort(importances)[::-1][:20]
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(range(20), importances[idx_top], color="olivedrab")
ax.set_xticks(range(20))
ax.set_xticklabels([f"F{i}" for i in idx_top], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Importance")
ax.set_title("Random Forest — Top 20 Feature Importances")
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=300); plt.close()

# Plot 6: Yield Distribution
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(y_te,       bins=30, alpha=0.6, color="steelblue",  label="Actual")
ax.hist(y_pred_fus, bins=30, alpha=0.6, color="darkorange", label="Predicted (Fusion)")
ax.set_xlabel("Yield (t/ha)"); ax.set_ylabel("Count")
ax.set_title("Yield Distribution — Actual vs Predicted"); ax.legend()
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/yield_distribution.png", dpi=300); plt.close()

# Plot 7: Per-crop RMSE
crop_names_map = {0:"Wheat-like", 1:"Rice-like", 2:"Maize-like"}
fig, ax = plt.subplots(figsize=(8,5))
for model_name, preds in [("Fusion",y_pred_fus),("RF",y_pred_rf),("LSTM",y_pred_lstm)]:
    crop_rmses = []
    for cls in [0,1,2]:
        mask = crop_labels[te_idx] == cls
        if mask.sum() > 0:
            crop_rmses.append(float(np.sqrt(np.mean((y_te[mask]-preds[mask])**2))))
        else:
            crop_rmses.append(0.0)
    ax.plot([crop_names_map[c] for c in [0,1,2]], crop_rmses, marker="o", label=model_name)
ax.set_ylabel("RMSE (t/ha)"); ax.set_title("Per-Crop RMSE Comparison"); ax.legend()
plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/per_crop_rmse.png", dpi=300); plt.close()

print("\nAll 7 plots saved to /content/plots/")

# ── Download everything ───────────────────────────────────────
import shutil
from google.colab import files

shutil.make_archive("/content/crop_yield_results", "zip", "/content/results")
shutil.make_archive("/content/crop_yield_models",  "zip", "/content/models")
shutil.make_archive("/content/crop_yield_plots",   "zip", "/content/plots")

files.download("/content/crop_yield_results.zip")
files.download("/content/crop_yield_models.zip")
files.download("/content/crop_yield_plots.zip")

print("\nDone! 3 zip files downloaded.")
