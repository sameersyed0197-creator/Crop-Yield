# AI-Based Crop Yield Prediction — CNN-LSTM Fusion

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Abstract

This project presents an end-to-end deep learning pipeline for crop yield prediction using a CNN-LSTM fusion architecture with an attention gate. The CNN branch extracts spatial-spectral features from multispectral image sequences, while the LSTM branch encodes weekly weather time-series. A learned attention gate weights each branch's contribution before the regression head. The system is evaluated against standalone CNN, standalone LSTM, Random Forest, and SVR baselines. Yield labels are derived from established FAO crop response functions applied to real soil and meteorological observations from the UCI Crop Recommendation dataset.

## Results

| Model             | RMSE   | MAE    | R²     | MAPE%  |
|-------------------|--------|--------|--------|--------|
| Standalone CNN    | 0.6828 | 0.5106 | 0.6895 | 28.41  |
| Standalone LSTM   | 0.4663 | 0.3586 | 0.8552 | 19.84  |
| Fusion (CNN+LSTM) | 0.4851 | 0.3742 | 0.8433 | 21.62  |
| Random Forest     | 0.5016 | 0.3768 | 0.8324 | 21.16  |
| SVR               | 0.6427 | 0.4720 | 0.7249 | 25.56  |

The standalone LSTM achieves the best R²=0.8552, demonstrating that temporal weather sequences are the dominant predictor of crop yield. The fusion model outperforms Random Forest and SVR baselines.

## Datasets

| File | Source | Description |
|------|--------|-------------|
| `crop_recommendation.csv` | UCI / Kaggle | 2200 rows, 22 crops, real soil + weather snapshots (N, P, K, pH, temperature, humidity, rainfall) |
| `daily_temp.csv` | jbrownlee/Datasets | 3650 days of real Melbourne daily min-temperature (1981–1990) |
| `monthly_sunspots.csv` | jbrownlee/Datasets | 2820 months of real sunspot counts → scaled to solar radiation proxy |

Yield labels are computed via FAO crop response functions from the real input features. Image patches (32×32, 7 bands) are derived analytically from NDVI/EVI computed from the weather and soil inputs.

## Project Structure

```
AGRI-YIELD/
├── run_pipeline.py                  # Lightning AI / local entry point
└── crop_yield_prediction/
    ├── config.py                    # All hyperparameters and paths
    ├── requirements.txt
    ├── data/
    │   └── raw/                     # Upload CSVs here
    │       ├── crop_recommendation.csv
    │       ├── daily_temp.csv
    │       └── monthly_sunspots.csv
    ├── src/
    │   ├── data_preprocessing.py    # Loads CSVs, builds .npy datasets
    │   ├── feature_engineering.py   # NDVI/EVI, scalers, ML flat features
    │   ├── cnn_model.py             # TimeDistributed CNN encoder
    │   ├── lstm_model.py            # Stacked LSTM encoder
    │   ├── fusion_model.py          # Attention-gated CNN-LSTM fusion
    │   ├── train.py                 # Trains all 5 models
    │   └── evaluate.py              # Metrics + 6 plots
    ├── dashboard/app.py             # Streamlit dashboard
    └── notebooks/
        ├── colab_cell2.py           # Self-contained Google Colab training cell
        └── EDA_and_Results.ipynb
```

## Quick Start — Lightning AI

1. Open a new Studio in [Lightning AI](https://lightning.ai)
2. Clone this repo in the terminal:
   ```bash
   git clone https://github.com/<your-username>/agri-yield.git
   cd agri-yield
   ```
3. Install dependencies:
   ```bash
   pip install -r crop_yield_prediction/requirements.txt
   ```
4. Run the full pipeline:
   ```bash
   python run_pipeline.py
   ```
   This downloads the 3 CSVs automatically, generates data, trains all 5 models, and saves results to `outputs/`.

5. (Optional) Launch the dashboard:
   ```bash
   streamlit run crop_yield_prediction/dashboard/app.py
   ```

## Quick Start — Google Colab

1. Upload `crop_recommendation.csv`, `daily_temp.csv`, `monthly_sunspots.csv` to `/content/`
2. Paste and run `notebooks/colab_cell2.py` — it is fully self-contained
3. Three zip files auto-download: results, models, plots

## Model Architecture

```
Image Input (T, 32, 32, 7)        Weather Input (T, 6)
        |                                  |
TimeDistributed(CNN Encoder)        Masking
  Conv2D(32) + BN + Pool            LSTM(128, return_seq=True)
  Conv2D(64) + BN + Pool            Dropout(0.2)
  Conv2D(128) + BN                  LSTM(64)
  GlobalAvgPool -> Dense(128)       Dropout(0.2)
        |                           Dense(64) -> wx_feat(64)
    LSTM(64)                               |
  sp_feat(64)                              |
        |                                  |
        +-------- Attention Gate ----------+
                  Dense(2, softmax)
                  [w_cnn, w_lstm]
                  sp_feat * w_cnn
                  wx_feat * w_lstm
                        |
                  Concat(128)
                  Dense(128, relu)
                  Dropout(0.2)
                  Dense(64, relu)
                  Dense(1, linear)
                        |
               Yield Prediction (t/ha)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| NUM_FIELDS | 2000 |
| SEQUENCE_LENGTH | 16 weeks |
| IMAGE_HEIGHT/WIDTH | 32×32 |
| NUM_BANDS | 5 (+2 NDVI/EVI = 7) |
| NUM_WEATHER_FEATURES | 6 |
| EPOCHS | 80 |
| BATCH_SIZE | 32 |
| LEARNING_RATE | 0.001 |
| DROPOUT_RATE | 0.2 |
| EARLY_STOPPING_PATIENCE | 20 |

## Tech Stack

- Keras 3.x + TensorFlow 2.16+
- scikit-learn (RandomForest, SVR)
- NumPy, Pandas, Matplotlib, Joblib
- Streamlit (dashboard)

## License

MIT License — free to use for academic and research purposes.
