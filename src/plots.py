# src/plots.py
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-GUI backend for servers/terminals
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_pred_vs_actual(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residuals_hist(residuals, out_path, bins=30):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=bins)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residuals_vs_fitted(y_pred, residuals, out_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted (Predicted)")
    plt.ylabel("Residual")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_importance(model, feature_names, out_path, max_features=25):
    # Works for tree-based models (RandomForest, XGBoost, etc.)
    if not hasattr(model, "feature_importances_"):
        return False
    importances = model.feature_importances_
    order = np.argsort(importances)[-max_features:]
    imp_vals = importances[order]
    imp_feats = np.array(feature_names)[order]

    plt.figure(figsize=(7, max(4, len(imp_feats) * 0.3)))
    plt.barh(range(len(imp_feats)), imp_vals)
    plt.yticks(range(len(imp_feats)), imp_feats)
    plt.xlabel("Importance")
    plt.title("Feature Importance (top {})".format(len(imp_feats)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True
