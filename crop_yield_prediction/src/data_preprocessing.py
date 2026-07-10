"""
Data preprocessing — generates spatially meaningful satellite image patches
with Gaussian NDVI blobs, stress circles, and yield labels directly tied
to both image (CNN) and weather (LSTM) signals.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, RAW_DIR, NUM_FIELDS, SEQUENCE_LENGTH, NUM_BANDS,
    NUM_WEATHER_FEATURES, IMAGE_HEIGHT, IMAGE_WIDTH, RANDOM_SEED,
    TEST_SPLIT, VALIDATION_SPLIT
)

SEASON_CURVE = [0.10, 0.15, 0.20, 0.28, 0.38, 0.52, 0.65,
                0.78, 0.85, 0.82, 0.72, 0.60, 0.48, 0.35, 0.22, 0.12]

CROP_CLASS_MAP = {
    "rice": 1, "maize": 2, "wheat": 0, "jute": 0,
    "cotton": 2, "coconut": 1, "papaya": 2, "orange": 0,
    "apple": 0, "muskmelon": 2, "watermelon": 2, "grapes": 0,
    "mango": 1, "banana": 2, "pomegranate": 0, "lentil": 0,
    "blackgram": 0, "mungbean": 0, "mothbeans": 0,
    "pigeonpeas": 0, "kidneybeans": 0, "chickpea": 0, "coffee": 1,
}
BASE_YIELDS = {0: 3.5, 1: 4.2, 2: 5.1}
CROP_NAMES  = {0: "wheat", 1: "rice", 2: "maize"}


def generate_field_patch(week, field_health, num_stress_patches, rng):
    """
    Generate one 32x32x7 image patch with spatially meaningful structure.
    field_health: 0.0–1.0 (directly correlated with yield).
    Bands: [Red, NIR, Blue, Green, SWIR, NDVI, EVI]
    """
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    season_mult = SEASON_CURVE[min(week, 15)]

    # Gaussian NDVI blob — healthy fields have large bright center blob
    base = np.zeros((H, W), dtype=np.float32)
    cx = rng.integers(H // 4, 3 * H // 4)
    cy = rng.integers(W // 4, 3 * W // 4)
    base[cx, cy] = 1.0
    sigma = 3.0 + field_health * 6.0      # healthy=wide blob, stressed=narrow
    blob = gaussian_filter(base, sigma=sigma)
    blob = blob / (blob.max() + 1e-8)
    ndvi_map = (blob * field_health * season_mult).astype(np.float32)

    # Stress patches — dark circles (pest / drought damage)
    for _ in range(num_stress_patches):
        sx = int(rng.integers(2, H - 6))
        sy = int(rng.integers(2, W - 6))
        radius = int(rng.integers(2, 6))
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - sx) ** 2 + (xx - sy) ** 2 <= radius ** 2
        ndvi_map[mask] *= 0.12

    # Spectral bands derived from NDVI map
    noise = lambda s: rng.normal(0, s, (H, W)).astype(np.float32)
    nir   = np.clip(ndvi_map * 0.6 + 0.1  + noise(0.02), 0, 1)
    red   = np.clip((1 - ndvi_map) * 0.15 + noise(0.01), 0, 1)
    blue  = np.clip(red * 0.7              + noise(0.01), 0, 1)
    green = np.clip((nir + red) * 0.4     + noise(0.01), 0, 1)
    swir  = np.clip(1 - ndvi_map * 0.8    + noise(0.02), 0, 1)
    evi   = np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8), -1, 1)

    patch = np.stack([red, nir, blue, green, swir,
                      np.clip(ndvi_map, 0, 1), evi.astype(np.float32)], axis=-1)
    return patch.astype(np.float32)


def generate_and_save():
    """Main pipeline: load real CSVs → build sequences → save .npy files."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load real crop data ───────────────────────────────────────────────
    print("[1/5] Loading crop_recommendation.csv ...")
    crop_rec = pd.read_csv(os.path.join(RAW_DIR, "crop_recommendation.csv"))
    print(f"  crop_recommendation: {crop_rec.shape}")

    repeats = int(np.ceil(NUM_FIELDS / len(crop_rec)))
    df = pd.concat([crop_rec] * repeats, ignore_index=True).sample(
        n=NUM_FIELDS, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"[2/5] Generating {NUM_FIELDS} fields ...")
    images      = np.zeros((NUM_FIELDS, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2), dtype=np.float32)
    weather     = np.zeros((NUM_FIELDS, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES), dtype=np.float32)
    yields      = np.zeros(NUM_FIELDS, dtype=np.float32)
    crop_labels = np.zeros(NUM_FIELDS, dtype=np.int32)
    ndvi_store  = np.zeros((NUM_FIELDS, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

    for i in range(NUM_FIELDS):
        row        = df.iloc[i]
        crop_class = CROP_CLASS_MAP.get(row["label"], 0)
        crop_labels[i] = crop_class

        # Field health — uniform 0.2–1.0, directly drives both image and yield
        field_health  = float(rng.uniform(0.2, 1.0))
        num_stress    = int(rng.integers(0, 7))          # 0=healthy, 6=very stressed

        # Real soil / weather snapshot
        base_temp     = float(row["temperature"])
        base_humidity = float(row["humidity"])
        base_rain     = float(row["rainfall"])

        # Weather correlated with field_health (healthy fields → better rain/humidity)
        rain_mean = base_rain * (0.5 + field_health * 0.8)
        for t in range(SEQUENCE_LENGTH):
            weather[i, t, 0] = float(np.clip(rain_mean  + rng.normal(0, rain_mean * 0.2), 0, rain_mean * 3))
            weather[i, t, 1] = float(base_temp + 5 + rng.normal(0, 2))
            weather[i, t, 2] = float(base_temp      + rng.normal(0, 1))
            weather[i, t, 3] = float(np.clip(base_humidity * (0.6 + field_health * 0.4) + rng.normal(0, 5), 20, 100))
            weather[i, t, 4] = float(np.clip(15 + rng.normal(0, 3), 5, 30))
            weather[i, t, 5] = float(np.clip(field_health * 0.4 + 0.1 + rng.normal(0, 0.05), 0.05, 0.6))

        # Image sequence
        for t in range(SEQUENCE_LENGTH):
            patch = generate_field_patch(t, field_health, num_stress, rng)
            images[i, t]    = patch
            ndvi_store[i, t] = patch[:, :, 5]

        # Yield — tied to BOTH image signal (field_health, num_stress) and weather
        total_rain = weather[i, :, 0].sum()
        peak_ndvi  = ndvi_store[i].mean(axis=(1, 2)).max()
        avg_temp   = weather[i, :, 1].mean()
        heat_stress = max(0.0, float(avg_temp) - 32.0)

        yield_val = (
            BASE_YIELDS[crop_class]
            + 1.2  * field_health          # spatial signal  → CNN learns
            + 0.8  * peak_ndvi             # NDVI signal     → CNN learns
            + 0.003 * total_rain           # rainfall signal → LSTM learns
            - 0.10 * heat_stress           # heat penalty    → LSTM learns
            - 0.18 * num_stress            # stress patches  → CNN learns
            + float(rng.normal(0, 0.12))   # small noise only
        )
        yields[i] = float(np.clip(yield_val, 0.5, 12.0))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{NUM_FIELDS} fields processed ...")

    print("[3/5] Saving .npy files ...")
    np.save(os.path.join(PROCESSED_DIR, "images.npy"),      images)
    np.save(os.path.join(PROCESSED_DIR, "weather.npy"),     weather)
    np.save(os.path.join(PROCESSED_DIR, "yields.npy"),      yields)
    np.save(os.path.join(PROCESSED_DIR, "crop_labels.npy"), crop_labels)
    np.save(os.path.join(PROCESSED_DIR, "ndvi.npy"),        ndvi_store)

    meta = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]].copy()
    meta["crop_class"] = crop_labels
    meta["yield_t_ha"] = yields
    meta.to_csv(os.path.join(PROCESSED_DIR, "field_metadata.csv"), index=False)

    print(f"[4/5] Yield stats: mean={yields.mean():.2f}  std={yields.std():.2f}"
          f"  min={yields.min():.2f}  max={yields.max():.2f}")
    print(f"[5/5] Done — images:{images.shape}  weather:{weather.shape}  yields:{yields.shape}")
    return images, weather, yields, crop_labels


def load_data():
    """Load preprocessed .npy files from disk."""
    images      = np.load(os.path.join(PROCESSED_DIR, "images.npy"))
    weather     = np.load(os.path.join(PROCESSED_DIR, "weather.npy"))
    yields      = np.load(os.path.join(PROCESSED_DIR, "yields.npy"))
    crop_labels = np.load(os.path.join(PROCESSED_DIR, "crop_labels.npy"))
    return images, weather, yields, crop_labels


def split_data(images, weather, yields, crop_labels):
    """Split into train / val / test."""
    rng = np.random.default_rng(RANDOM_SEED)
    N   = len(yields)
    idx = rng.permutation(N)

    n_test = int(N * TEST_SPLIT)
    n_val  = int(N * VALIDATION_SPLIT)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    splits = {}
    for name, i in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = {
            "images":      images[i],
            "weather":     weather[i],
            "yields":      yields[i],
            "crop_labels": crop_labels[i],
        }
    print(f"Split — train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
    return splits


if __name__ == "__main__":
    generate_and_save()
