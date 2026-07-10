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
    """Slices a single column from a 2-D tensor: output = input[:, index:index+1]."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index:self.index + 1]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


def build_fusion_model(image_input_shape, weather_input_shape):
    """
    Attention-gated CNN-LSTM fusion with residual connection and L2 regularization.
    image_input_shape:   (T, H, W, C)
    weather_input_shape: (T, NUM_WEATHER_FEATURES)
    """
    reg = regularizers.L2(L2)

    # --- Branch 1: CNN spatial encoder over time ---
    image_inp = Input(shape=image_input_shape, name="image_input")
    cnn_enc = build_cnn_encoder(image_input_shape[1:])
    spatial_seq = layers.TimeDistributed(cnn_enc, name="td_cnn")(image_inp)   # (T, 128)
    spatial_feat = layers.LSTM(64, return_sequences=False, name="spatial_lstm")(spatial_seq)  # (64,)
    sp_proj = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="cnn_proj")(spatial_feat)

    # --- Branch 2: LSTM weather encoder ---
    weather_inp = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc = build_lstm_encoder(weather_input_shape)
    weather_feat = lstm_enc(weather_inp)                                        # (64,)
    wx_proj = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="lstm_proj")(weather_feat)

    # --- Attention-weighted fusion ---
    concat_gate = layers.Concatenate(name="gate_input")([sp_proj, wx_proj])    # (128,)
    attn_weights = layers.Dense(2, activation="softmax", name="attn_weights")(concat_gate)  # (2,)
    cnn_w  = SliceChannel(0, name="cnn_weight")(attn_weights)                  # (1,)
    lstm_w = SliceChannel(1, name="lstm_weight")(attn_weights)                 # (1,)
    cnn_scaled  = layers.Multiply(name="cnn_scaled")([sp_proj,  cnn_w])
    lstm_scaled = layers.Multiply(name="lstm_scaled")([wx_proj, lstm_w])
    fused = layers.Concatenate(name="fusion")([cnn_scaled, lstm_scaled])       # (128,)

    # --- Deep regression head: 512 → 256 → 128 → 1 ---
    x = layers.Dense(512, activation="relu", kernel_regularizer=reg)(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # --- Residual: add CNN branch directly into 128-unit layer ---
    cnn_res = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="cnn_residual")(sp_proj)
    x128 = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Add(name="residual_add")([x128, cnn_res])
    x = layers.Dropout(DROPOUT_RATE)(x)

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
    wx_shape = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_fusion_model(img_shape, wx_shape)
