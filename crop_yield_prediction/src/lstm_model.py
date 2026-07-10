import os
import sys
import keras
from keras import layers, Model, Input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_UNITS, DROPOUT_RATE, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES


def build_lstm_encoder(input_shape):
    """
    LSTM temporal encoder.
    input_shape: (T, NUM_WEATHER_FEATURES)
    Returns: Keras Model outputting (64,) feature vector.
    """
    inp = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(LSTM_UNITS[0], return_sequences=True)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.LSTM(LSTM_UNITS[1], return_sequences=False)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(64, activation="relu")(x)

    model = Model(inp, out, name="lstm_encoder")
    model.summary()
    return model


def build_standalone_lstm(weather_input_shape):
    """
    Full LSTM model with regression head.
    weather_input_shape: (T, NUM_WEATHER_FEATURES)
    """
    inp = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc = build_lstm_encoder(weather_input_shape)
    x = lstm_enc(inp)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inp, out, name="standalone_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()
    return model


if __name__ == "__main__":
    shape = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_standalone_lstm(shape)
