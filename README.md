# Pune House Price Prediction

An end-to-end regression project predicting apartment/villa prices across
Pune localities: BHK, area, age, floor, amenities → price (in Lakhs INR).

## ⚠️ Important: data is synthetic, not real transactions

There is no reliable, freely downloadable transaction-level dataset for
Pune real estate — several "Pune house price" files circulating online
are actually the Bangalore dataset with locality names swapped (I checked;
this one was too, so I didn't use it).

Instead, this project generates a synthetic dataset where each locality's
base price-per-sqft is drawn from **real, sourced 2026 market-rate
benchmarks** (NoBroker, 99acres, Pune real-estate reports), then combined
with a documented pricing model (age, floor, amenities, property type,
BHK, distance to IT parks) plus realistic noise. Full methodology and
sources are in `DATA_SOURCING.md` — read that before treating any number
here as a real price.

**Good for**: practicing a full ML pipeline on realistic, India-specific
real-estate feature relationships.
**Not for**: actual property valuation or investment decisions.

## Project structure

```
pune_house_price_project/
├── DATA_SOURCING.md          # methodology + real benchmark sources
├── data/
│   └── pune_housing.csv       # generated synthetic dataset (12,000 rows)
├── src/
│   ├── 00_generate_data.py    # generates the calibrated synthetic dataset
│   ├── 01_eda.py               # exploratory analysis
│   ├── 02_preprocessing.py    # cleaning, feature engineering, pipeline
│   ├── 03_train_models.py     # trains + compares 6 models, tunes the best
│   └── 04_evaluate.py         # diagnostic plots, feature importance
├── models/
│   ├── preprocessor.pkl        # fitted sklearn ColumnTransformer (joblib)
│   ├── xgb_model.json          # tuned XGBoost model (native JSON - version-safe)
│   ├── model_type.json
│   ├── final_metrics.json
│   └── model_comparison.csv
├── plots/                     # EDA + evaluation charts
├── app.py                     # Streamlit interactive predictor
└── requirements.txt
```

## How to run it

```bash
pip install -r requirements.txt

cd src
python 00_generate_data.py    # generate the calibrated synthetic dataset
python 01_eda.py              # explore it, generates plots/
python 02_preprocessing.py    # clean + build preprocessing pipeline
python 03_train_models.py     # train, compare, tune models
python 04_evaluate.py         # diagnostic plots for the final model

cd ..
streamlit run app.py          # launch the interactive predictor
```

## The app

`app.py` is a polished, dark-themed Streamlit UI with three tabs:
- **Predict Price** — form for property details, styled result card,
  a gauge showing where the prediction sits within its zone's price
  range, and comparison badges (vs. zone average, vs. city average).
- **Market Insights** — interactive Plotly charts: median price by
  locality, price distribution by zone, price vs. sqft colored by BHK,
  and a dark-mode map of localities sized by median price.
- **About This Model** — performance metrics and the full data-sourcing
  disclaimer, so it's never mistaken for a real valuation tool.

Requires `plotly` in addition to the base requirements (already in
`requirements.txt`).

## Localities covered (30 total, grouped by zone)

| Zone | Localities | ₹/sqft benchmark (2026) |
|---|---|---|
| Premium West | Baner, Aundh, Kalyani Nagar, Koregaon Park, Boat Club Road, Pashan | 12,000–22,000 |
| IT Corridor | Kharadi, Viman Nagar, Kothrud, Balewadi | 9,000–16,000 |
| Mid-tier | Hinjewadi, Wakad, Bavdhan, Ravet, Magarpatta | 7,000–10,000 |
| Central | Camp, Narayan Peth, Somwar Peth, Swargate, Shivaji Nagar | 6,000–9,000 |
| Outskirts | Wagholi, Moshi, Undri, Narhe, Kondhwa, Hadapsar, Chinchwad, Kiwale, NIBM Road, Sinhgad Road | 5,000–8,000 |

## Results

| Model | CV RMSE (Lakhs) |
|---|---|
| Linear Regression | 35.19 |
| Ridge Regression | 35.19 |
| Decision Tree | 36.69 |
| Random Forest | 28.80 |
| Gradient Boosting | 26.28 |
| **XGBoost** | **25.74** |

**Final tuned model (XGBoost) on the held-out test set:**
- **RMSE**: ₹23.8 lakhs
- **MAE**: ₹16.5 lakhs
- **R²**: 0.926

**Top predictive features**: zone (premium/IT-corridor/outskirts), total
sqft, BHK, property type (villa premium), and age.

**Sanity check**: a 3BHK/1500sqft apartment predicts ~₹16,500/sqft in
Baner and ~₹6,600/sqft in Wagholi — both land squarely inside the
sourced real-world benchmark ranges for those localities.

## A note on model loading

The preprocessor and model are saved and loaded **separately**, not as one
pickled sklearn Pipeline:
- `preprocessor.pkl` — the sklearn `ColumnTransformer`, saved with `joblib`.
- `xgb_model.json` — the tuned XGBoost model, saved with XGBoost's own
  `.save_model()` (native JSON format).

This is deliberate: XGBoost's pickled binary buffer is not reliably
compatible across different xgboost versions/machines, and pickling it
inside a joblib-dumped Pipeline caused an `XGBoostError: input stream
corrupted` when loaded on a machine with a different xgboost version than
the one used to train it. XGBoost's native JSON format doesn't have this
problem. If you retrain the model, `03_train_models.py` already saves it
this way — no extra steps needed. `app.py` and `04_evaluate.py` load both
pieces and chain `preprocessor.transform()` → `xgb_model.predict()`
manually instead of calling `.predict()` on a single pipeline object.

## Possible extensions

- Replace the synthetic generator with a real dataset if you can source
  one (e.g., a scraped/licensed 99acres or MagicBricks export).
- Add more granular location signal — distance to metro stations,
  specific IT park (Hinjewadi vs EON Kharadi), school proximity.
- Add SHAP explainability to the Streamlit app.
