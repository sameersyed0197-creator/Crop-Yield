import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_WEATHER_FEATURES,
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, MODEL_DIR, RESULTS_DIR, RANDOM_SEED
)
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import prepare_features
from src.cnn_model import build_standalone_cnn
from src.lstm_model import build_standalone_lstm
from src.fusion_model import build_fusion_model
from src.utils import set_seeds, create_output_dirs, log_experiment


def get_callbacks(model_name):
    """Return standard Keras training callbacks."""
    ckpt_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
    return [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]


def train_deep_model(model, X_train, y_train, X_val, y_val, model_name):
    """Train a Keras model and return history dict."""
    save_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    if os.path.exists(save_path):
        print(f"[SKIP] {model_name} already trained at {save_path}")
        return {}
    print(f"\n{'='*50}\nTraining {model_name}...\n{'='*50}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    model.save(save_path)
    print(f"Saved {model_name} to {MODEL_DIR}")
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def train_sklearn_model(model, X_train, y_train, model_name):
    """Train a scikit-learn model and save with joblib."""
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if os.path.exists(save_path):
        print(f"[SKIP] {model_name} already trained at {save_path}")
        return
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Saved {model_name} to {MODEL_DIR}")


def main():
    set_seeds(RANDOM_SEED)
    create_output_dirs()

    print("\n[Step 1] Loading data...")
    images, weather, yields, crop_labels = load_data()
    splits = split_data(images, weather, yields, crop_labels)

    print("\n[Step 2] Preparing features...")
    img_tr, wx_tr, ml_tr, _, img_sc, wx_sc = prepare_features(
        splits["train"]["images"], splits["train"]["weather"], fit_scalers=True
    )

    def scale_split(split):
        from src.feature_engineering import (
            normalize_images, normalize_weather,
            compute_vegetation_indices, extract_temporal_stats, create_ml_features
        )
        img_n, _ = normalize_images(split["images"], fit=False, scaler=img_sc)
        wx_n, _  = normalize_weather(split["weather"], fit=False, scaler=wx_sc)
        ndvi, _  = compute_vegetation_indices(split["images"])
        stats    = extract_temporal_stats(ndvi)
        ml       = create_ml_features(stats, wx_n)
        return img_n, wx_n, ml

    img_val, wx_val, ml_val = scale_split(splits["val"])
    img_te,  wx_te,  ml_te  = scale_split(splits["test"])

    y_tr  = splits["train"]["yields"]
    y_val = splits["val"]["yields"]

    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)

    # Load existing histories if present
    hist_path = os.path.join(RESULTS_DIR, "training_histories.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            histories = json.load(f)
    else:
        histories = {}

    # 1. Standalone CNN
    cnn_model = build_standalone_cnn(img_shape)
    h = train_deep_model(cnn_model, img_tr, y_tr, img_val, y_val, "standalone_cnn")
    if h:
        histories["standalone_cnn"] = h

    # 2. Standalone LSTM
    lstm_model = build_standalone_lstm(wx_shape)
    h = train_deep_model(lstm_model, wx_tr, y_tr, wx_val, y_val, "standalone_lstm")
    if h:
        histories["standalone_lstm"] = h

    # 3. Fusion model
    fusion = build_fusion_model(img_shape, wx_shape)
    h = train_deep_model(
        fusion,
        [img_tr, wx_tr], y_tr,
        [img_val, wx_val], y_val,
        "fusion_model"
    )
    if h:
        histories["fusion_model"] = h

    # 4. Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    train_sklearn_model(rf, ml_tr, y_tr, "random_forest")

    # 5. SVR
    svr = SVR(kernel="rbf", C=10, epsilon=0.1)
    train_sklearn_model(svr, ml_tr, y_tr, "svr")

    # Save histories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"\nTraining histories saved to {hist_path}")

    log_experiment(
        {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": RANDOM_SEED},
        {"models_trained": list(histories.keys())}
    )
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
