# src/evaluate.py
# Evaluate predictions from all_predictions.csv (or any predictions file)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(pred_csv: str = "all_predictions.csv",
                         target_col: str = "Delivery_Time",
                         pred_col: str = "Predicted_Delivery_Time",
                         plots_dir: str = "plots/evaluation"):
    path = Path(pred_csv)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_csv}")

    df = pd.read_csv(path)

    if target_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"File must have '{target_col}' and '{pred_col}' columns")

    y_true = df[target_col].values
    y_pred = df[pred_col].values
    residuals = y_true - y_pred

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("📊 Evaluation Metrics")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    # Save plots
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect Prediction")
    plt.xlabel("Actual Delivery Time")
    plt.ylabel("Predicted Delivery Time")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.savefig(plots_path / "pred_vs_actual_eval.png")
    plt.close()

    # Residual histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    plt.title("Residuals Histogram")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.savefig(plots_path / "residuals_hist_eval.png")
    plt.close()

    # Residuals vs Fitted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Delivery Time")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.savefig(plots_path / "residuals_vs_fitted_eval.png")
    plt.close()

    print(f"✅ Plots saved to {plots_path.resolve()}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    evaluate_predictions()
