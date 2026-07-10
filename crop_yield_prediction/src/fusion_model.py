import os
import sys
import keras
from keras import layers, Model, Input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DROPOUT_RATE, LEARNING_RATE, SEQUENCE_LENGTH,
    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_WEATHER_FEATURES,
    PLOT_DIR, MODEL_DIR
)
from src.cnn_model import build_cnn_encoder
from src.lstm_model import build_lstm_encoder


def build_fusion_model(image_input_shape, weather_input_shape):
    """
    CNN-LSTM fusion model.
    image_input_shape:   (T, H, W, C)
    weather_input_shape: (T, NUM_WEATHER_FEATURES)
    Returns: compiled Keras Model predicting yield (tons/ha).
    """
    # --- Branch 1: CNN spatial encoder over time ---
    image_inp = Input(shape=image_input_shape, name="image_input")
    cnn_enc = build_cnn_encoder(image_input_shape[1:])
    spatial_seq = layers.TimeDistributed(cnn_enc, name="td_cnn")(image_inp)  # (T, 128)
    spatial_feat = layers.LSTM(64, return_sequences=False, name="spatial_lstm")(spatial_seq)  # (64,)

    # --- Branch 2: LSTM weather encoder ---
    weather_inp = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc = build_lstm_encoder(weather_input_shape)
    weather_feat = lstm_enc(weather_inp)  # (64,)

    # --- Fusion ---
    fused = layers.Concatenate(name="fusion")([spatial_feat, weather_feat])  # (128,)
    x = layers.Dense(256, activation="relu")(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inputs=[image_inp, weather_inp], outputs=out, name="fusion_model")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    # Save architecture diagram
    os.makedirs(PLOT_DIR, exist_ok=True)
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(PLOT_DIR, "fusion_architecture.png"),
            show_shapes=True, dpi=150
        )
        print(f"Architecture diagram saved to {PLOT_DIR}/fusion_architecture.png")
    except Exception:
        pass  # pydot/graphviz optional

    return model


if __name__ == "__main__":
    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_fusion_model(img_shape, wx_shape)
