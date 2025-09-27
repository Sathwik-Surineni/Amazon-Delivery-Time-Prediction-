import numpy as np
import pandas as pd


def _combine_date_time(date_series, time_series):
    date_str = pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%d")
    time_str = pd.to_datetime(
        time_series, format="%H:%M:%S", errors="coerce"
    ).dt.time.astype(str)
    return pd.to_datetime(date_str + " " + time_str, errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Distances
    df["distance_km"] = haversine_km(
        df["Store_Latitude"],
        df["Store_Longitude"],
        df["Drop_Latitude"],
        df["Drop_Longitude"],
    )

    # Time features
    order_dt = _combine_date_time(df["Order_Date"], df["Order_Time"])
    pickup_dt = _combine_date_time(df["Order_Date"], df["Pickup_Time"])

    df["pickup_delay_min"] = (pickup_dt - order_dt).dt.total_seconds() / 60.0
    df["order_hour"] = pd.to_datetime(
        df["Order_Time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour
    df["order_dow"] = pd.to_datetime(
        df["Order_Date"], errors="coerce"
    ).dt.dayofweek  # 0=Mon

    # Cyclical hour
    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24.0)

    # Drop raw date/time (optional)
    for c in ["Order_Time", "Pickup_Time", "Order_Date"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    return df


CATEGORICAL = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
