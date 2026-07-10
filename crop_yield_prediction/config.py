import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

NUM_FIELDS = 2000
SEQUENCE_LENGTH = 16
NUM_BANDS = 5
NUM_WEATHER_FEATURES = 6
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

CNN_FILTERS = [32, 64, 128]
LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.2
DENSE_UNITS = [256, 128]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 80
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15

RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 15
