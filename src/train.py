"""
Amazon Delivery Time Prediction Training Script (Python 3.11+)

- Engineers compact features: order_hour, order_dow, pickup_delay, distance_km
- Drops raw datetime columns (so sklearn won't choke on datetime dtype)
- Handles categoricals with OrdinalEncoder (memory safe, not one-hot)
- Trains RF, GBR, Linear models; compares metrics; logs to MLflow
- Saves best model to models/best/model.joblib
- Creates plots in reports/: residuals, pred-vs-actual, feature importances
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# Use a non-GUI backend so scripts don't try to open windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# -------------------------
# Config
# -------------------------
DATA_CANDIDATES = [
    "data/amazon_delivery.csv",
    "data/processed/train.csv",
    "data/processed/dataset.csv",
    "data/train.csv",
    "data/dataset.csv",
    "dataset.csv",
]
TARGET_CANDIDATES = ["Delivery_Time", "delivery_time", "target", "DeliveryTime", "y"]

KNOWN_SMALL_CATS = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

MODEL_DIR = Path("models/best")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.joblib"

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# Helpers
# -------------------------
def find_dataset() -> Path:
    for p in DATA_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find dataset. Looked for: {DATA_CANDIDATES}")

def get_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found. Expected one of: {TARGET_CANDIDATES}")

def haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Order timestamp -> order_hour, order_dow
    if "Order_Date" in df.columns and "Order_Time" in df.columns:
        order_dt = pd.to_datetime(
            df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str),
            errors="coerce",
        )
        df["order_hour"] = order_dt.dt.hour
        df["order_dow"] = order_dt.dt.dayofweek
    else:
        df["order_hour"] = np.nan
        df["order_dow"] = np.nan

    # Pickup delay (min)
    if "Pickup_Time" in df.columns and "Order_Date" in df.columns and "Order_Time" in df.columns:
        order_dt = pd.to_datetime(
            df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str),
            errors="coerce",
        )
        pickup_dt = pd.to_datetime(df["Pickup_Time"], errors="coerce")
        df["pickup_delay"] = (pickup_dt - order_dt).dt.total_seconds() / 60.0
    else:
        df["pickup_delay"] = np.nan

    # Geo features (accept alt names)
    alt_map = {
        "Store_Latitude": "Store_Lat",
        "Store_Longitude": "Store_Long",
        "Drop_Latitude": "Drop_Lat",
        "Drop_Longitude": "Drop_Long",
    }
    for src, dst in alt_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    if {"Store_Lat", "Store_Long", "Drop_Lat", "Drop_Long"}.issubset(df.columns):
        df["distance_km"] = haversine(
            df["Store_Lat"], df["Store_Long"], df["Drop_Lat"], df["Drop_Long"]
        )
    else:
        df["distance_km"] = np.nan

    # Drop raw datetime cols
    for col in ["Order_Date", "Order_Time", "Pickup_Time"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df

def to_float32(X):
    return X.astype(np.float32)

def build_column_sets(X: pd.DataFrame):
    numeric_mask = X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    num_cols = X.columns[numeric_mask].tolist()

    cat_cols = []
    for c in X.columns[~numeric_mask]:
        if X[c].nunique(dropna=True) <= 30:
            cat_cols.append(c)

    for c in KNOWN_SMALL_CATS:
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)

    num_cols = [c for c in num_cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("to_float32", FunctionTransformer(to_float32)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

# -------------------------
# Plotting helpers
# -------------------------
def _save_residual_plot(residuals, name):
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=40)
    plt.title(f"Residual Distribution: {name}")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name}_residuals.png")
    plt.close()

def _save_pred_vs_actual(y_true, y_pred, name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual: {name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name}_pred_vs_actual.png")
    plt.close()

def _save_feature_importance(pipe, num_cols, cat_cols, name):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    feat_names = list(num_cols) + list(cat_cols)
    importances = model.feature_importances_
    if len(importances) != len(feat_names):
        feat_names = [f"f{i}" for i in range(len(importances))]
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order], y=[feat_names[i] for i in order])
    plt.title(f"Top 20 Features: {name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name}_feature_importance.png")
    plt.close()

def evaluate_and_plot(pipe, num_cols, cat_cols, X_valid, y_valid, name):
    y_pred = pipe.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"{name} -> {metrics}")
    residuals = y_valid - y_pred
    _save_residual_plot(residuals, name)
    _save_pred_vs_actual(y_valid, y_pred, name)
    _save_feature_importance(pipe, num_cols, cat_cols, name)
    return metrics

# -------------------------
# Main
# -------------------------
def main():
    data_path = find_dataset()
    df_raw = pd.read_csv(data_path)

    target = get_target_column(df_raw)
    y = df_raw[target]
    X_raw = df_raw.drop(columns=[target])

    X = engineer_features(X_raw)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    num_cols, cat_cols = build_column_sets(X_train)
    pre = build_preprocessor(num_cols, cat_cols)

    candidates = {
        "rf": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "linreg": LinearRegression(),
    }

    mlflow.set_experiment("amazon_delivery_time")
    best_name, best_model, best_metrics = None, None, None

    for name, est in candidates.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline([("pre", pre), ("model", est)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            metrics = evaluate_and_plot(pipe, num_cols, cat_cols, X_valid, y_valid, name)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                best_name, best_model, best_metrics = name, pipe, metrics

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model: {best_name} RMSE: {best_metrics['rmse']:.3f}")
    print(f"Saved: {MODEL_DIR}")

if __name__ == "__main__":
    main()
