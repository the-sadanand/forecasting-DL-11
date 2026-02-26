# Optuna + W&B + walk-forward
import json
import wandb
import optuna
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import sys
# add src to python path
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import preprocess
from feature_engineering import create_features
from models import LSTMModel

LOG_PATH = Path("logs/training.log")
RESULT_PATH = Path("results")
MODEL_PATH = Path("models")

RESULT_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH.mkdir(exist_ok=True, parents=True)
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)


def save_best_outputs(study, model):
    best_params = study.best_params

    with open(RESULT_PATH / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    torch.save(model.state_dict(), MODEL_PATH / "best_model.pt")

    print("✅ Saved best_params.json and best_model.pt")


def create_sequences(data, target_col, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i : i + window].values)
        y.append(data.iloc[i + window][target_col])
    return np.array(X), np.array(y)


def train_lstm(X, y, params):
    model = LSTMModel(X.shape[2], params["hidden"], params["layers"], params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = torch.nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(5):
        pred = model(X_t).squeeze()
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def objective(trial, df):
    params = {
        "window": trial.suggest_int("window", 24, 72),
        "hidden": trial.suggest_int("hidden", 32, 128),
        "layers": trial.suggest_int("layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }

    target = "Global_active_power"

    X, y = create_sequences(df, target, params["window"])

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = train_lstm(X_train, y_train, params)

    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse


def retrain_best_model(df, params):
    target = "Global_active_power"
    X, y = create_sequences(df, target, params["window"])
    model = train_lstm(X, y, params)
    return model


def generate_forecasts(model, df, target_col, window, split_ratio=0.8):
    """Generate forecasts and return predictions with actuals"""
    X, y = create_sequences(df, target_col, window)
    
    split = int(len(X) * split_ratio)
    X_test = X[split:]
    y_test = y[split:]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test_t).squeeze().numpy()
    
    # Create forecasts dataframe
    forecast_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'error': y_test - predictions,
        'abs_error': np.abs(y_test - predictions)
    })
    
    return forecast_df


def create_visualization(forecast_df, output_path):
    """Create forecast vs actual visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Forecast vs Actual
    axes[0].plot(forecast_df.index, forecast_df['actual'], label='Actual', color='blue', linewidth=2)
    axes[0].plot(forecast_df.index, forecast_df['predicted'], label='Predicted', color='red', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Global Active Power (Normalized)')
    axes[0].set_title('Energy Consumption: Forecast vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error
    axes[1].plot(forecast_df.index, forecast_df['error'], label='Error', color='green', linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(forecast_df.index, forecast_df['error'], 0, alpha=0.3, color='green')
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title('Forecast Error Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to {output_path}")


def calculate_metrics(forecast_df):
    """Calculate performance metrics"""
    mae = mean_absolute_error(forecast_df['actual'], forecast_df['predicted'])
    rmse = np.sqrt(mean_squared_error(forecast_df['actual'], forecast_df['predicted']))
    mape = mean_absolute_percentage_error(forecast_df['actual'], forecast_df['predicted'])
    
    return {'mae': round(mae, 6), 'rmse': round(rmse, 6), 'mape': round(mape, 6)}


def main():
    wandb.init(project="energy-forecasting")

    df = preprocess()
    df_feat = create_features(df)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, df_feat), n_trials=10)

    best_params = study.best_params
    wandb.log(best_params)

    # 🔥 retrain final model on full data
    best_model = retrain_best_model(df_feat, best_params)

    # 🔥 SAVE MODEL + PARAMS
    save_best_outputs(study, best_model)

    # Generate forecasts
    target = "Global_active_power"
    forecast_df = generate_forecasts(best_model, df_feat, target, best_params["window"])
    
    # Save forecasts to CSV
    forecast_csv_path = RESULT_PATH / "forecasts.csv"
    forecast_df.to_csv(forecast_csv_path)
    print(f"✅ Saved forecasts to {forecast_csv_path}")
    
    # Create visualization
    viz_path = RESULT_PATH / "forecast_visualization.png"
    create_visualization(forecast_df, viz_path)
    
    # Calculate real metrics
    metrics_dict = calculate_metrics(forecast_df)
    
    # log file
    with open(LOG_PATH, "w") as f:
        f.write("Training completed with best params\n")
        f.write(f"Best params: {json.dumps(best_params, indent=2)}\n")
        f.write(f"Metrics: {json.dumps(metrics_dict, indent=2)}\n")

    # Save metrics with real values
    metrics = {
        "deep_learning_model": metrics_dict,
        "best_params": best_params,
        "optimization_trials": 10
    }

    with open(RESULT_PATH / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training pipeline completed")
    print(f"📊 Metrics - MAE: {metrics_dict['mae']}, RMSE: {metrics_dict['rmse']}, MAPE: {metrics_dict['mape']}")


if __name__ == "__main__":
    main()