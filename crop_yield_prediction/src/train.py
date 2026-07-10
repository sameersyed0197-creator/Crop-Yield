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
from src.utils import set_seeds, create_output_dirs, log_experiment
from keras import layers, Model, Input
import keras


@keras.saving.register_keras_serializable()
class SliceChannel(layers.Layer):
    """Slices a single column from a 2-D tensor: output = input[:, index:index+1]."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index:self.index + 1]

    def get_config(self):
        return {**super().get_config(), "index": self.index}



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

    # 3. Fusion model — attention-gated, two-phase training
    fusion_path = os.path.join(MODEL_DIR, "fusion_model.keras")
    if os.path.exists(fusion_path):
        print(f"[SKIP] fusion_model already trained at {fusion_path}")
    else:
        print(f"\n{'='*50}\nTraining fusion_model...\n{'='*50}")
        T, H, W, C = img_shape
        _, F = wx_shape

        def _cnn_enc():
            inp = Input(shape=(H, W, C))
            x = layers.Conv2D(32, 3, padding="same")(inp)
            x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D(2)(x)
            x = layers.Conv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D(2)(x)
            x = layers.Conv2D(128, 3, padding="same")(x)
            x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
            x = layers.GlobalAveragePooling2D()(x)
            return Model(inp, layers.Dense(128, activation="relu")(x), name="cnn_enc")

        from config import DROPOUT_RATE, LEARNING_RATE
        img_inp = Input(shape=img_shape, name="image_input")
        sp_seq  = layers.TimeDistributed(_cnn_enc(), name="td_cnn")(img_inp)
        sp_feat = layers.LSTM(64, name="spatial_lstm")(sp_seq)
        sp_proj = layers.Dense(64, activation="relu", name="cnn_proj")(sp_feat)

        wx_inp  = Input(shape=wx_shape, name="weather_input")
        x2      = layers.Masking(mask_value=0.0)(wx_inp)
        x2      = layers.LSTM(128, return_sequences=True)(x2)
        x2      = layers.Dropout(DROPOUT_RATE)(x2)
        x2      = layers.LSTM(64)(x2)
        x2      = layers.Dropout(DROPOUT_RATE)(x2)
        wx_feat = layers.Dense(64, activation="relu", name="lstm_proj")(x2)

        gate_in = layers.Concatenate(name="gate_input")([sp_proj, wx_feat])
        gate    = layers.Dense(2, activation="softmax", name="branch_gate")(gate_in)
        cnn_w   = SliceChannel(0, name="cnn_weight")(gate)
        lstm_w  = SliceChannel(1, name="lstm_weight")(gate)
        fused   = layers.Concatenate(name="fusion")(
            [layers.Multiply(name="cnn_scaled")([sp_proj, cnn_w]),
             layers.Multiply(name="lstm_scaled")([wx_feat, lstm_w])]
        )
        x = layers.Dense(128, activation="relu")(fused)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(1, activation="linear", name="yield_output")(x)
        fusion = Model(inputs=[img_inp, wx_inp], outputs=out, name="fusion_model")

        # Phase 1: freeze CNN branch, warm up LSTM + gate
        for layer in fusion.layers:
            if layer.name in ("td_cnn", "spatial_lstm", "cnn_proj"):
                layer.trainable = False
        fusion.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
        print("Phase 1: warming up LSTM branch (CNN frozen)...")
        fusion.fit([img_tr, wx_tr], y_tr, validation_data=([img_val, wx_val], y_val),
                   epochs=20, batch_size=BATCH_SIZE, verbose=1)

        # Phase 2: unfreeze all, joint fine-tune at lower LR
        for layer in fusion.layers:
            layer.trainable = True
        fusion.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE * 0.3), loss="mse", metrics=["mae"])
        print("Phase 2: joint fine-tuning (all layers unfrozen)...")
        h = fusion.fit([img_tr, wx_tr], y_tr, validation_data=([img_val, wx_val], y_val),
                       epochs=EPOCHS, batch_size=BATCH_SIZE,
                       callbacks=get_callbacks("fusion_model"), verbose=1)
        fusion.save(fusion_path)
        histories["fusion_model"] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

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
