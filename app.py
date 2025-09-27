# app.py
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from datetime import datetime, date
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Amazon Delivery Time", layout="wide")

# ------------------------
# Paths
# ------------------------
MODEL_PATH = Path("models/best/model.joblib")
RAW_DATA = Path("data/amazon_delivery.csv")
REPORTS_DIR = Path("reports")

# ------------------------
# Unpickle helper
# ------------------------
# Your trained pipeline includes FunctionTransformer(to_float32).
# Defining it here lets joblib find it when unpickling.
def to_float32(X):
    return X.astype(np.float32)

# ------------------------
# Utils to match train.py
# ------------------------
def _normalize_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Store_Latitude": "Store_Lat",
        "Store_Longitude": "Store_Long",
        "Drop_Latitude": "Drop_Lat",
        "Drop_Longitude": "Drop_Long",
    }
    out = df.copy()
    for src, dest in mapping.items():
        if src in out.columns and dest not in out.columns:
            out[dest] = out[src]
    return out

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors train.py: creates order_hour, order_dow, pickup_delay (min), distance_km
    and drops raw datetime columns.
    """
    df = _normalize_geo_columns(df_in).copy()

    # order_hour, order_dow
    if "Order_Date" in df.columns and "Order_Time" in df.columns:
        order_dt = pd.to_datetime(
            df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str),
            errors="coerce",
        )
        df["order_hour"] = order_dt.dt.hour
        df["order_dow"] = order_dt.dt.dayofweek
    else:
        now = datetime.now()
        df["order_hour"] = now.hour
        df["order_dow"] = now.weekday()

    # pickup_delay (minutes)
    if {"Pickup_Time", "Order_Date", "Order_Time"}.issubset(df.columns):
        order_dt = pd.to_datetime(
            df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str),
            errors="coerce",
        )
        pickup_dt = pd.to_datetime(df["Pickup_Time"], errors="coerce")
        df["pickup_delay"] = ((pickup_dt - order_dt).dt.total_seconds() / 60).fillna(0)
    else:
        df["pickup_delay"] = 0.0

    # distance_km
    if {"Store_Lat", "Store_Long", "Drop_Lat", "Drop_Long"}.issubset(df.columns):
        df["distance_km"] = df.apply(
            lambda r: haversine(r["Store_Lat"], r["Store_Long"], r["Drop_Lat"], r["Drop_Long"]),
            axis=1,
        )
    else:
        df["distance_km"] = 0.0

    # Drop raw datetime cols
    for col in ["Order_Date", "Order_Time", "Pickup_Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# ------------------------
# Caching
# ------------------------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        # to_float32 must be in this module's globals for unpickling
        return joblib.load(MODEL_PATH)
    st.warning("Model not found. Please train first:  python -m src.train")
    return None

@st.cache_data
def load_data(n: int = 5000) -> pd.DataFrame:
    if RAW_DATA.exists():
        df = pd.read_csv(RAW_DATA)
        return df.sample(min(n, len(df)), random_state=42)
    return pd.DataFrame()

model = load_model()

# ------------------------
# UI
# ------------------------
st.title("📦 Amazon Delivery Time Prediction")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "📊 Explore", "📑 Batch Predict", "📈 Model Insights"]
)

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Single Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        agent_age = st.number_input("Agent Age", 18, 70, 30)
        agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
        weather = st.selectbox(
            "Weather",
            ["Sunny", "Windy", "Stormy", "Fog", "Cloudy", "Sandstorms", "Snowy", "Rainy"],
        )
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])

    with col2:
        vehicle = st.selectbox("Vehicle", ["Bike", "Scooter", "Car", "Truck"])
        area = st.selectbox("Area", ["Urban", "Suburban", "Rural"])
        category = st.selectbox(
            "Category",
            [
                "Electronics", "Books", "Jewelry", "Toys", "Skincare", "Snacks",
                "Outdoors", "Apparel", "Sports", "Grocery", "Pet Supplies", "Home",
                "Cosmetics", "Kitchen", "Clothing", "Shoes",
            ],
        )
        store_lat = st.number_input("Store Latitude", -90.0, 90.0, 17.3850, step=0.0001, format="%.6f")
        store_lon = st.number_input("Store Longitude", -180.0, 180.0, 78.4867, step=0.0001, format="%.6f")

    with col3:
        drop_lat = st.number_input("Drop Latitude", -90.0, 90.0, 17.4500, step=0.0001, format="%.6f")
        drop_lon = st.number_input("Drop Longitude", -180.0, 180.0, 78.4000, step=0.0001, format="%.6f")
        order_date = st.date_input("Order Date", value=date.today())
        order_time = st.time_input("Order Time", value=datetime.now().time())
        pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

    if st.button("Predict Delivery Time", use_container_width=True):
        # Build one-row DataFrame with original column names (as in your dataset)
        row = pd.DataFrame([{
            "Order_ID": "demo",
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": store_lat,
            "Store_Longitude": store_lon,
            "Drop_Latitude": drop_lat,
            "Drop_Longitude": drop_lon,
            "Order_Date": str(order_date),
            "Order_Time": order_time.strftime("%H:%M:%S"),
            # train.py accepted both "HH:MM:SS" or full datetime in Pickup_Time. We pass a full string safely:
            "Pickup_Time": f"{order_date} {pickup_time.strftime('%H:%M:%S')}",
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category,
        }])

        X = add_engineered_features(row)
        if model is None:
            st.stop()
        try:
            pred = float(model.predict(X)[0])
            # If your target is hours (most likely from your logs):
            st.success(f"Estimated Delivery Time: **{pred:.2f} hours**")
            # If your target were minutes, you'd display both:
            # st.success(f"Estimated Delivery Time: **{pred:.1f} min** (~{pred/60:.2f} hours)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Tab 2: Explore ---
with tab2:
    st.subheader("Quick Explore")
    df = load_data()
    if df.empty:
        st.info("Place amazon_delivery.csv in data/ and reload.")
    else:
        st.write("Sample of data:")
        st.dataframe(df.head(20), use_container_width=True)

        eng = add_engineered_features(df)
        left, right = st.columns(2)
        if "order_hour" in eng.columns:
            with left:
                st.bar_chart(eng["order_hour"].value_counts().sort_index())
                st.caption("Orders by Hour of Day")
        if "order_dow" in eng.columns:
            with right:
                st.bar_chart(eng["order_dow"].value_counts().sort_index())
                st.caption("Orders by Day of Week (0=Mon)")

        if "distance_km" in eng.columns:
            st.line_chart(eng[["distance_km"]].reset_index(drop=True))
            st.caption("Distance distribution (sample)")

# --- Tab 3: Batch Predict ---
with tab3:
    st.subheader("Batch Predict (upload CSV with original columns)")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up and model:
        try:
            up_df = pd.read_csv(up)
            eng = add_engineered_features(up_df)
            preds = model.predict(eng)
            out = up_df.copy()
            out["Predicted_Delivery_Time"] = preds
            st.success(f"Predicted {len(out)} rows.")
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# --- Tab 4: Model Insights ---
with tab4:
    st.subheader("Model Insights (from training)")
    if REPORTS_DIR.exists():
        imgs = sorted(REPORTS_DIR.glob("*.png"))
        if imgs:
            for img in imgs:
                st.image(str(img), caption=img.name, use_column_width=True)
        else:
            st.info("No report images found. Run training to generate plots.")
    else:
        st.info("No reports/ folder found. Run training to generate plots.")
