import os
import sys
import keras
from keras import layers, Model, Input, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DROPOUT_RATE, LEARNING_RATE, SEQUENCE_LENGTH,
    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_WEATHER_FEATURES,
    PLOT_DIR, MODEL_DIR
)
from src.cnn_model import build_cnn_encoder
from src.lstm_model import build_lstm_encoder

L2 = 0.001


@keras.saving.register_keras_serializable()
class SliceChannel(layers.Layer):
    """Kept for backward-compat with any saved models that used it."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index:self.index + 1]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


def build_fusion_model(image_input_shape, weather_input_shape):
    """
    Feature-wise attention-gated CNN-LSTM fusion.
    - 64-dim sigmoid gates per branch (not 2-scalar softmax)
    - Dual residual from both branches
    - Deep head: 256 → 128 → 64 → 1
    - L2=0.001 on all Dense layers
    image_input_shape:   (T, H, W, C)
    weather_input_shape: (T, F)
    """
    reg = regularizers.L2(L2)

    # ── Branch 1: CNN spatial encoder ────────────────────────────────────
    image_inp   = Input(shape=image_input_shape, name="image_input")
    cnn_enc     = build_cnn_encoder(image_input_shape[1:])
    spatial_seq = layers.TimeDistributed(cnn_enc, name="td_cnn")(image_inp)        # (T, 128)
    spatial_feat = layers.LSTM(64, return_sequences=False, name="spatial_lstm")(spatial_seq)
    sp_proj     = layers.Dense(64, activation="relu",
                               kernel_regularizer=reg, name="cnn_proj")(spatial_feat)

    # ── Branch 2: LSTM weather encoder ───────────────────────────────────
    weather_inp  = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc     = build_lstm_encoder(weather_input_shape)
    weather_feat = lstm_enc(weather_inp)                                            # (64,)
    wx_proj      = layers.Dense(64, activation="relu",
                                kernel_regularizer=reg, name="lstm_proj")(weather_feat)

    # ── Feature-wise cross-attention (64-dim gates, not 2-scalar) ────────
    concat_all = layers.Concatenate(name="gate_input")([sp_proj, wx_proj])         # (128,)

    cnn_gate     = layers.Dense(64, activation="sigmoid",
                                kernel_regularizer=reg, name="cnn_gate")(concat_all)
    cnn_attended = layers.Multiply(name="cnn_attended")([sp_proj, cnn_gate])       # (64,)

    lstm_gate     = layers.Dense(64, activation="sigmoid",
                                 kernel_regularizer=reg, name="lstm_gate")(concat_all)
    lstm_attended = layers.Multiply(name="lstm_attended")([wx_proj, lstm_gate])    # (64,)

    fused = layers.Concatenate(name="fusion")([cnn_attended, lstm_attended])       # (128,)

    # ── Dual residual from both branches ─────────────────────────────────
    residual = layers.Dense(128, activation="relu",
                            kernel_regularizer=reg, name="residual_proj")(concat_all)
    fused_res = layers.Add(name="residual_add")([fused, residual])                 # (128,)

    # ── Regression head: 256 → 128 → 64 → 1 ─────────────────────────────
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(fused_res)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64,  activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inputs=[image_inp, weather_inp], outputs=out, name="fusion_model")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    os.makedirs(PLOT_DIR, exist_ok=True)
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(PLOT_DIR, "fusion_architecture.png"),
            show_shapes=True, dpi=150
        )
    except Exception:
        pass

    return model


if __name__ == "__main__":
    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_fusion_model(img_shape, wx_shape)
