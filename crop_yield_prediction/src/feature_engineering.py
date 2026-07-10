import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, RANDOM_SEED


def compute_vegetation_indices(image_sequence):
    """Compute NDVI and EVI from image sequence (N, T, H, W, bands)."""
    red = image_sequence[..., 0]
    nir = image_sequence[..., 1]
    blue = image_sequence[..., 2]
    ndvi = (nir - red) / (nir + red + 1e-8)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
    return ndvi.astype(np.float32), evi.astype(np.float32)


def extract_temporal_stats(ndvi_sequence):
    """
    Extract temporal statistics from NDVI sequence.
    ndvi_sequence: (N, T, H, W) → returns (N, 5) feature array.
    """
    ndvi_mean_spatial = ndvi_sequence.mean(axis=(2, 3))  # (N, T)
    peak_ndvi = ndvi_mean_spatial.max(axis=1)
    mean_ndvi = ndvi_mean_spatial.mean(axis=1)

    T = ndvi_mean_spatial.shape[1]
    t_axis = np.arange(T, dtype=np.float32)
    ndvi_slope = np.array([
        np.polyfit(t_axis, ndvi_mean_spatial[i], 1)[0]
        for i in range(len(ndvi_mean_spatial))
    ], dtype=np.float32)

    green_up_week = ndvi_mean_spatial.argmax(axis=1).astype(np.float32)

    half_peak = peak_ndvi / 2
    senescence_week = np.array([
        next((t for t in range(int(green_up_week[i]), T) if ndvi_mean_spatial[i, t] < half_peak[i]),
             T - 1)
        for i in range(len(ndvi_mean_spatial))
    ], dtype=np.float32)

    return np.stack([peak_ndvi, mean_ndvi, ndvi_slope, green_up_week, senescence_week], axis=1)


def normalize_weather(weather_data, fit=True, scaler=None):
    """StandardScaler normalization for weather features."""
    N, T, F = weather_data.shape
    flat = weather_data.reshape(-1, F)
    if fit:
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "weather_scaler.pkl"))
    else:
        flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(N, T, F).astype(np.float32), scaler


def normalize_images(image_data, fit=True, scaler=None):
    """MinMaxScaler normalization per band for image data."""
    N, T, H, W, C = image_data.shape
    flat = image_data.reshape(-1, C)
    if fit:
        scaler = MinMaxScaler()
        flat_scaled = scaler.fit_transform(flat)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "image_scaler.pkl"))
    else:
        flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(N, T, H, W, C).astype(np.float32), scaler


def create_ml_features(ndvi_stats, weather_data):
    """
    Create flat feature vector for baseline ML models.
    ndvi_stats: (N, 5), weather_data: (N, T, F) → (N, 5 + T*F)
    """
    N = len(ndvi_stats)
    weather_flat = weather_data.reshape(N, -1)
    return np.concatenate([ndvi_stats, weather_flat], axis=1).astype(np.float32)


def prepare_features(images, weather, fit_scalers=True):
    """Full feature preparation pipeline."""
    ndvi, evi = compute_vegetation_indices(images)
    ndvi_stats = extract_temporal_stats(ndvi)
    images_norm, img_scaler = normalize_images(images, fit=fit_scalers)
    weather_norm, wx_scaler = normalize_weather(weather, fit=fit_scalers)
    ml_features = create_ml_features(ndvi_stats, weather_norm)
    return images_norm, weather_norm, ml_features, ndvi_stats, img_scaler, wx_scaler
