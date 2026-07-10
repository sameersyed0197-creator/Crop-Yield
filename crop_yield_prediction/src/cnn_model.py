import os
import sys
import keras
from keras import layers, Model, Input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_FILTERS, DROPOUT_RATE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS


def build_cnn_encoder(input_shape):
    """
    CNN spatial feature extractor for a single timestep.
    input_shape: (H, W, C)
    Returns: Keras Model outputting (128,) feature vector.
    """
    inp = Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS[0], 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(CNN_FILTERS[1], 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(CNN_FILTERS[2], 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(128, activation="relu")(x)

    model = Model(inp, out, name="cnn_encoder")
    model.summary()
    return model


def build_standalone_cnn(image_input_shape):
    """
    Full CNN model with TimeDistributed wrapper + LSTM + regression head.
    image_input_shape: (T, H, W, C)
    """
    inp = Input(shape=image_input_shape, name="image_input")
    cnn_enc = build_cnn_encoder(image_input_shape[1:])
    x = layers.TimeDistributed(cnn_enc)(inp)          # (T, 128)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inp, out, name="standalone_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()
    return model


if __name__ == "__main__":
    shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    build_standalone_cnn(shape)
