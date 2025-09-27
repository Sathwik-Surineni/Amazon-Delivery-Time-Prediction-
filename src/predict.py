# src/predict.py
from pathlib import Path
import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

MODEL_PATH = Path("models/best/model.joblib")
DATA_PATH = Path("data/amazon_delivery.csv")  # change if needed
OUTPUT_PATH = Path("all_predictions.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    from math import radians as rad
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = sin(dlat/2) ** 2 + cos(rad(lat1)) * cos(rad(lat2)) * sin(dlon/2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def engineer_features(df: pd.DataFrame):
    out = df.copy()

    if "Order_Date" in out.columns and "Order_Time" in out.columns:
        order_dt = pd.to_datetime(out["Order_Date"].astype(str) + " " + out["Order_Time"].astype(str),
                                  errors="coerce")
        out["order_hour"] = order_dt.dt.hour
        out["order_dow"] = order_dt.dt.dayofweek

    if "Pickup_Time" in out.columns and "Order_Date" in out.columns and "Order_Time" in out.columns:
        order_dt = pd.to_datetime(out["Order_Date"].astype(str) + " " + out["Order_Time"].astype(str),
                                  errors="coerce")
        pickup_dt = pd.to_datetime(out["Pickup_Time"], errors="coerce")
        out["pickup_delay"] = ((pickup_dt - order_dt).dt.total_seconds() / 60).fillna(0)

    alt_map = {
        "Store_Latitude": "Store_Lat",
        "Store_Longitude": "Store_Long",
        "Drop_Latitude": "Drop_Lat",
        "Drop_Longitude": "Drop_Long",
    }
    for src, dst in alt_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    if {"Store_Lat", "Store_Long", "Drop_Lat", "Drop_Long"}.issubset(out.columns):
        out["distance_km"] = out.apply(
            lambda r: haversine(r["Store_Lat"], r["Store_Long"], r["Drop_Lat"], r["Drop_Long"]),
            axis=1,
        )

    for col in ["Order_Date", "Order_Time", "Pickup_Time"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out

def predict_on_dataset():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    model = joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    target_col = next((c for c in ["Delivery_Time","delivery_time","target","DeliveryTime","y"] if c in df.columns), None)
    X = df.drop(columns=[target_col]) if target_col else df

    X_feat = engineer_features(X)

    preds = model.predict(X_feat)

    out = df.copy()
    out["Predicted_Delivery_Time"] = preds
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    predict_on_dataset()
