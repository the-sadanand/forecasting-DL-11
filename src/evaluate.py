import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import LSTMModel
from preprocess import preprocess
from feature_engineering import create_features


def create_sequences(data, target_col, window):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i : i + window].values)
        y.append(data.iloc[i + window][target_col])
    return np.array(X), np.array(y)


def evaluate():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    model_path = Path("models/best_model.pt")
    
    if not model_path.exists():
        print("❌ Model not found. Please run train.py first.")
        return

    # Load data
    df = preprocess()
    df_feat = create_features(df)

    # Load best params
    with open(results_dir / "best_params.json") as f:
        params = json.load(f)

    target = "Global_active_power"
    X, y = create_sequences(df_feat, target, params["window"])

    # Split data (80/20 train/test)
    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]

    # Load model
    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=params["hidden"],
        num_layers=params["layers"],
        dropout=params["dropout"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate predictions
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_t).squeeze().numpy()

    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)

    metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "actual": y_test,
        "prediction": preds,
        "error": y_test - preds
    })
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    # Plot forecast
    plt.figure(figsize=(14, 6))
    plt.plot(y_test[:200], label="Actual", linewidth=2)
    plt.plot(preds[:200], label="Prediction", linewidth=2, alpha=0.7)
    plt.legend()
    plt.title("Forecast vs Actual (First 200 samples)")
    plt.xlabel("Time Index")
    plt.ylabel("Global Active Power (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "forecast.png", dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ Evaluation complete — outputs saved in results/")


if __name__ == "__main__":
    evaluate()