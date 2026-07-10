import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_WEATHER_FEATURES,
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, MODEL_DIR, RESULTS_DIR, RANDOM_SEED,
    LEARNING_RATE
)
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import (
    prepare_features, normalize_images, normalize_weather,
    compute_vegetation_indices, extract_temporal_stats, create_ml_features
)
from src.cnn_model import build_standalone_cnn
from src.lstm_model import build_standalone_lstm
from src.fusion_model import build_fusion_model
from src.utils import set_seeds, create_output_dirs, log_experiment, compute_metrics


def cosine_schedule(total_epochs, lr_start, lr_min=1e-6):
    def schedule(epoch, _lr):
        cos = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return float(lr_min + (lr_start - lr_min) * cos)
    return schedule


def get_callbacks(model_name, lr_sched=None, patience=None):
    ckpt = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
    cbs = [
        EarlyStopping(monitor="val_loss",
                      patience=patience or EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
    ]
    if lr_sched:
        cbs.append(keras.callbacks.LearningRateScheduler(lr_sched, verbose=0))
    return cbs


def train_deep_model(model, X_tr, y_tr, X_val, y_val, model_name, lr_start=None):
    save_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    if os.path.exists(save_path):
        print(f"[SKIP] {model_name} already trained.")
        return {}
    print(f"\n{'='*50}\nTraining {model_name}...\n{'='*50}")
    lr_s = cosine_schedule(EPOCHS, lr_start or LEARNING_RATE) if lr_start else None
    h = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=get_callbacks(model_name, lr_s), verbose=1)
    model.save(save_path)
    print(f"Saved {model_name}")
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def train_sklearn_model(model, X_tr, y_tr, model_name):
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if os.path.exists(save_path):
        print(f"[SKIP] {model_name} already trained.")
        return
    print(f"\nTraining {model_name}...")
    model.fit(X_tr, y_tr)
    joblib.dump(model, save_path)
    print(f"Saved {model_name}")


def train_fusion_two_phase(img_tr, wx_tr, y_tr, img_val, wx_val, y_val,
                           img_shape, wx_shape, histories):
    fusion_path = os.path.join(MODEL_DIR, "fusion_model.keras")
    if os.path.exists(fusion_path):
        print("[SKIP] fusion_model already trained.")
        return

    print(f"\n{'='*50}\nTraining fusion_model (two-phase)\n{'='*50}")
    fusion = build_fusion_model(img_shape, wx_shape)

    # Phase 1: freeze td_cnn, warm LSTM + attention + head (30 ep, lr=0.001)
    for layer in fusion.layers:
        if "td_cnn" in layer.name:
            layer.trainable = False
    fusion.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    print("\nPhase 1 (30 ep): CNN frozen ...")
    fusion.fit(
        [img_tr, wx_tr], y_tr,
        validation_data=([img_val, wx_val], y_val),
        epochs=30, batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True, verbose=1),
                   keras.callbacks.LearningRateScheduler(
                       cosine_schedule(30, 0.001), verbose=0)],
        verbose=1,
    )

    # Phase 2: unfreeze all, joint fine-tune (70 ep, lr=0.0001)
    for layer in fusion.layers:
        layer.trainable = True
    fusion.compile(optimizer=keras.optimizers.Adam(0.0001), loss="mse", metrics=["mae"])
    print("\nPhase 2 (70 ep): all layers unfrozen ...")
    h = fusion.fit(
        [img_tr, wx_tr], y_tr,
        validation_data=([img_val, wx_val], y_val),
        epochs=70, batch_size=BATCH_SIZE,
        callbacks=get_callbacks("fusion_model",
                                lr_sched=cosine_schedule(70, 0.0001),
                                patience=15),
        verbose=1,
    )
    fusion.save(fusion_path)
    print("Saved fusion_model")
    histories["fusion_model"] = {k: [float(v) for v in vals] for k, vals in h.history.items()}


def main():
    set_seeds(RANDOM_SEED)
    create_output_dirs()

    # ── Load full dataset ─────────────────────────────────────────────────
    print("\n[Step 1] Loading data...")
    images, weather, yields, crop_labels = load_data()
    print(f"  Yield stats — min:{yields.min():.2f}  max:{yields.max():.2f}  mean:{yields.mean():.2f}  std:{yields.std():.2f}")

    # ── Fit ALL scalers on FULL dataset before splitting ──────────────────
    # This prevents train-subset range mismatch in MinMaxScaler / StandardScaler
    print("\n[Step 2] Fitting scalers on full dataset ...")
    _, _, _, _, img_sc, wx_sc = prepare_features(images, weather, fit_scalers=True)

    # Yield scaler — fit on full yields, save for inverse-transform in evaluate
    yield_sc = StandardScaler()
    yield_sc.fit(yields.reshape(-1, 1))
    joblib.dump(yield_sc, os.path.join(MODEL_DIR, "yield_scaler.pkl"))
    print(f"  Yield scaler saved. Scaled range: "
          f"{yield_sc.transform([[yields.min()]])[0,0]:.2f} – "
          f"{yield_sc.transform([[yields.max()]])[0,0]:.2f}")

    # ── Split ─────────────────────────────────────────────────────────────
    splits = split_data(images, weather, yields, crop_labels)

    # ── Apply scalers to each split ───────────────────────────────────────
    def scale_split(split):
        img_n, _ = normalize_images(split["images"],  fit=False, scaler=img_sc)
        wx_n,  _ = normalize_weather(split["weather"], fit=False, scaler=wx_sc)
        ndvi, _  = compute_vegetation_indices(split["images"])
        stats    = extract_temporal_stats(ndvi)
        ml       = create_ml_features(stats, wx_n)
        # Scale yields
        y_sc = yield_sc.transform(split["yields"].reshape(-1, 1)).flatten().astype(np.float32)
        return img_n, wx_n, ml, y_sc

    img_tr,  wx_tr,  ml_tr,  y_tr  = scale_split(splits["train"])
    img_val, wx_val, ml_val, y_val = scale_split(splits["val"])
    img_te,  wx_te,  ml_te,  y_te  = scale_split(splits["test"])

    # Raw (unscaled) test yields for final metrics
    y_te_raw = splits["test"]["yields"]

    print(f"  Train yields (scaled): min={y_tr.min():.2f}  max={y_tr.max():.2f}")
    print(f"  Test  yields (scaled): min={y_te.min():.2f}  max={y_te.max():.2f}")

    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)

    hist_path = os.path.join(RESULTS_DIR, "training_histories.json")
    histories = json.load(open(hist_path)) if os.path.exists(hist_path) else {}

    # 1. Standalone CNN
    cnn_model = build_standalone_cnn(img_shape)
    h = train_deep_model(cnn_model, img_tr, y_tr, img_val, y_val,
                         "standalone_cnn", lr_start=LEARNING_RATE)
    if h:
        histories["standalone_cnn"] = h

    # 2. Standalone LSTM
    lstm_model = build_standalone_lstm(wx_shape)
    h = train_deep_model(lstm_model, wx_tr, y_tr, wx_val, y_val,
                         "standalone_lstm", lr_start=LEARNING_RATE)
    if h:
        histories["standalone_lstm"] = h

    # 3. Fusion — two-phase
    train_fusion_two_phase(img_tr, wx_tr, y_tr, img_val, wx_val, y_val,
                           img_shape, wx_shape, histories)

    # 4. Random Forest (trained on scaled yields)
    rf = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                               random_state=RANDOM_SEED, n_jobs=-1)
    train_sklearn_model(rf, ml_tr, y_tr, "random_forest")

    # 5. SVR (trained on scaled yields)
    svr = SVR(kernel="rbf", C=10, epsilon=0.1)
    train_sklearn_model(svr, ml_tr, y_tr, "svr")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)

    log_experiment(
        {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "seed": RANDOM_SEED},
        {"models_trained": list(histories.keys())}
    )

    # ── Final comparison table (inverse-transform predictions back to t/ha) ──
    print("\n[Final] Comparison table on held-out test set (original t/ha scale):")
    print("=" * 62)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8}")
    print("-" * 62)

    import pandas as pd
    results = {}
    order = ["standalone_cnn", "standalone_lstm", "fusion_model", "random_forest", "svr"]
    for name in order:
        path_k = os.path.join(MODEL_DIR, f"{name}.keras")
        path_p = os.path.join(MODEL_DIR, f"{name}.pkl")
        try:
            if os.path.exists(path_k):
                m = keras.models.load_model(path_k, safe_mode=False)
                if name == "fusion_model":
                    pred_sc = m.predict([img_te, wx_te], verbose=0).flatten()
                elif name == "standalone_cnn":
                    pred_sc = m.predict(img_te, verbose=0).flatten()
                else:
                    pred_sc = m.predict(wx_te, verbose=0).flatten()
            elif os.path.exists(path_p):
                m = joblib.load(path_p)
                pred_sc = m.predict(ml_te)
            else:
                continue
            # Inverse-transform to original t/ha scale
            pred_orig = yield_sc.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            metrics = compute_metrics(y_te_raw, pred_orig)
            results[name] = metrics
            print(f"{name:<22} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} "
                  f"{metrics['R2']:>8.4f} {metrics['MAPE']:>8.2f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")
    print("=" * 62)

    df_res = pd.DataFrame(results).T
    df_res.index.name = "Model"
    df_res.to_csv(os.path.join(RESULTS_DIR, "all_model_results.csv"))
    print(f"\nResults saved to {RESULTS_DIR}/all_model_results.csv")
    print("All models trained successfully.")


if __name__ == "__main__":
    main()
