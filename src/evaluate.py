import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.model import LSTMModel
from src.data_loader import load_data, create_sequences


def evaluate():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    processed_path = "data/processed/processed.csv"
    model_path = "models/best_model.pt"

    df = load_data(processed_path)

    with open("results/best_params.json") as f:
        params = json.load(f)

    X, y = create_sequences(df["Global_active_power"].values, params["window"])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(
        input_size=1,
        hidden_size=params["hidden"],
        num_layers=params["layers"],
        dropout=params["dropout"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(X).squeeze().numpy()

    y_true = y.numpy()

    mse = ((preds - y_true) ** 2).mean()
    mae = abs(preds - y_true).mean()

    metrics = {"mse": float(mse), "mae": float(mae)}

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "actual": y_true,
        "prediction": preds
    })
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    # Plot forecast
    plt.figure()
    plt.plot(y_true[:200], label="Actual")
    plt.plot(preds[:200], label="Prediction")
    plt.legend()
    plt.title("Forecast vs Actual")
    plt.savefig(results_dir / "forecast.png")
    plt.close()

    print("✅ Evaluation complete — outputs saved in results/")


if __name__ == "__main__":
    evaluate()