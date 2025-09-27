# 📦 Amazon Delivery Time Prediction

Predicting e-commerce delivery times using Machine Learning + Streamlit
A complete end-to-end ML project: data preprocessing, feature engineering, model training, experiment tracking, visualization, and deployment as a web app.

## 🌟 Overview

Delivery time is one of the most critical factors in customer satisfaction for e-commerce platforms like Amazon.
This project builds a predictive machine learning system that estimates delivery times (in hours) based on factors such as:

Order details (date, time, pickup delay)

Location coordinates (store & drop)

Environmental factors (traffic, weather, area)

Agent details (age, rating, vehicle type)

The project includes:

✔️ Clean data preprocessing pipeline
✔️ Feature engineering (time, geospatial, distance)
✔️ Multiple ML models trained & compared
✔️ Automatic evaluation & visualization (plots, feature importance)
✔️ Interactive Streamlit app for single & batch predictions

## 🛠️ Tech Stack

### Languages & Libraries
~~~
Python 3.11

Pandas, NumPy

Scikit-Learn, Joblib

Matplotlib, Seaborn

Streamlit (UI)

MLflow (experiment tracking)
~~~

### Project Workflow
~~~
Data: data/amazon_delivery.csv

Training: src/train.py

Prediction: src/predict.py

App: app.py
~~~
## 📂 Project Structure
~~~
Amazon-Delivery-Time-Prediction/
│
├── data/                 # Dataset (csv files)
├── models/               # Trained models (ignored in GitHub, generated locally)
├── reports/              # Metrics, plots (residuals, feature importance)
├── src/                  # Core ML code
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Batch prediction
│   └── features.py       # Feature engineering
├── app.py                # Streamlit app
├── requirements.txt      # Dependencies
├── .gitignore            # Ignore virtual envs, models, cache
└── README.md             # Project report (this file)
~~~
## 🔍 Features

### Feature Engineering
~~~
Pickup delay (minutes)

Distance using Haversine formula

Order hour & day of week
~~~

### Models Compared
~~~
Random Forest Regressor 🌲

Gradient Boosting Regressor 🌟

Linear Regression 📈
~~~
### Evaluation Metrics
~~~
RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score
~~~
## 📊 Model Results (Sample Run)
~~~
Model	RMSE ↓	MAE ↓	R² ↑
RandomForest	23.03	17.87	0.80
GradientBoost	24.35	19.21	0.77
LinearReg	45.03	34.79	0.23    
~~~
✅ Random Forest chosen as the best model.

## 🚀 How to Run
### 1️⃣ Clone Repository
~~~
git clone https://github.com/Sathwik-Surineni/Amazon-Delivery-Time-Prediction-.git
cd Amazon-Delivery-Time-Prediction-
~~~
### 2️⃣ Create Virtual Environment
~~~
python -m venv .venv
.venv\Scripts\activate   # On Windows
~~~
### 3️⃣ Install Dependencies
~~~
pip install -r requirements.txt

~~~

### 4️⃣ Train Models
~~~
python -m src.train
~~~
➡️ Saves best model in models/best/
### 5️⃣ Run Batch Prediction
~~~
python -m src.predict
~~~
### 6️⃣ Launch Streamlit App
~~~
streamlit run app.py
~~~
➡️ Opens an interactive web app for single/batch predictions.
##  Sample Streamlit Screens
~~~
Single Prediction Form (enter order, traffic, weather, etc.)

Exploration Dashboard (orders by hour/day, distance distribution)

Batch Prediction Upload (upload CSV & download predictions)
~~~
## 🔮 Future Improvements

-Train on larger dataset with deep learning (LSTMs for time series)

-Integrate real-time traffic & weather APIs

-Deploy app to Streamlit Cloud / AWS / Azure

-Add CI/CD pipeline for automated retraining
### ✨ This project demonstrates a real-world ML pipeline, from data to deployment. Designed for scalability and practical use in e-commerce analytics.
