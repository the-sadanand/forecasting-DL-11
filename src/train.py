# Optuna + W&B + Walk-Forward Validation + Prophet Baseline + Early Stopping
import json
import os
import sys
import logging
import warnings

# Suppress verbose logging from dependencies
logging.basicConfig(level=logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('pystan').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

import wandb
import optuna
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
from datetime import datetime

# add src to python path
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import preprocess
from feature_engineering import create_features
from models import LSTMModel

LOG_PATH = Path("logs/training.log")
RESULT_PATH = Path("results")
MODEL_PATH = Path("models")
MONITORING_PATH = Path("results/monitoring")

RESULT_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH.mkdir(exist_ok=True, parents=True)
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
MONITORING_PATH.mkdir(exist_ok=True, parents=True)


class EarlyStopping:
    """Early stopping callback for preventing overfitting"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class PerformanceMonitor:
    """Track training and validation metrics"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'epoch': []
        }

    def update(self, epoch, train_loss, val_loss, val_mae, val_rmse):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_mae'].append(val_mae)
        self.history['val_rmse'].append(val_rmse)

    def save(self, path):
        """Save monitoring history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, path):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['epoch'], self.history['val_rmse'], label='Val RMSE')
        axes[1].plot(self.history['epoch'], self.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Validation Metrics Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()


def save_best_outputs(study, model):
    best_params = study.best_params

    with open(RESULT_PATH / "best_params.json", "w", encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)

    torch.save(model.state_dict(), MODEL_PATH / "best_model.pt")

    print("[OK] Saved best_params.json and best_model.pt")


def create_sequences(data, target_col, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i : i + window].values)
        y.append(data.iloc[i + window][target_col])
    return np.array(X), np.array(y)


def train_lstm_with_early_stopping(X_train, y_train, X_val, y_val, params, epochs=20):
    """Train LSTM with early stopping and monitoring"""
    model = LSTMModel(X_train.shape[2], params["hidden"], params["layers"], params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = torch.nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    monitor = PerformanceMonitor()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(epochs):
        # Training
        model.train()
        pred = model(X_train_t).squeeze()
        train_loss = loss_fn(pred, y_train_t)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze().numpy()
            val_loss = loss_fn(model(X_val_t).squeeze(), y_val_t)
            val_mae = mean_absolute_error(y_val_t.numpy(), val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_t.numpy(), val_pred))

        monitor.update(epoch, train_loss.item(), val_loss.item(), val_mae, val_rmse)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        # Early stopping
        early_stopping(val_loss.item())
        if early_stopping.early_stop:
            print(f"  [WARN] Early stopping at epoch {epoch+1}")
            break

    return model, monitor


def train_lstm(X, y, params):
    """Original LSTM training (for compatibility)"""
    model = LSTMModel(X.shape[2], params["hidden"], params["layers"], params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = torch.nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    # Use batch training to reduce memory usage
    batch_size = 32
    num_epochs = 5

    for epoch in range(num_epochs):
        for i in range(0, len(X_t), batch_size):
            batch_X = X_t[i:i+batch_size]
            batch_y = y_t[i:i+batch_size]
            
            pred = model(batch_X).squeeze()
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def objective(trial, df):
    params = {
        "window": trial.suggest_int("window", 12, 36),  # Reduced from 72 to 36
        "hidden": trial.suggest_int("hidden", 32, 64),  # Reduced from 128 to 64
        "layers": trial.suggest_int("layers", 1, 2),    # Reduced from 3 to 2
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),  # Tighter range
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }

    target = "Global_active_power"
    
    # Use only last 10k samples to manage memory
    df_sample = df.tail(10000) if len(df) > 10000 else df

    X, y = create_sequences(df_sample, target, params["window"])

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = train_lstm(X_train, y_train, params)

    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse


def walk_forward_validation(df, params, n_folds=3):
    """Perform walk-forward validation for time-series data"""
    print(f"\n[*] Starting Walk-Forward Validation with {n_folds} folds...")
    
    target = "Global_active_power"
    X, y = create_sequences(df, target, params["window"])
    
    fold_results = []
    total_len = len(X)
    fold_size = total_len // (n_folds + 1)  # Reserve last fold for final test
    
    for fold in range(n_folds):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        
        # Walk-forward split
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = train_end + fold_size
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        
        # Train model
        model, monitor = train_lstm_with_early_stopping(X_train, y_train, X_val, y_val, params, epochs=10)
        
        # Evaluate
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()
        
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mape = mean_absolute_percentage_error(y_val, val_pred)
        
        fold_results.append({'mae': mae, 'rmse': rmse, 'mape': mape})
        print(f"  [OK] Fold {fold + 1} - MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
    
    # Average results
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_mape = np.mean([r['mape'] for r in fold_results])
    
    print(f"\n[*] Walk-Forward Average Metrics:")
    print(f"  MAE: {avg_mae:.6f}, RMSE: {avg_rmse:.6f}, MAPE: {avg_mape:.6f}")
    
    return {
        'fold_results': fold_results,
        'average_mae': avg_mae,
        'average_rmse': avg_rmse,
        'average_mape': avg_mape
    }


def train_prophet_baseline(df):
    """Train Prophet baseline model for comparison"""
    print("\n[*] Training Prophet Baseline...")
    
    try:
        # Prepare data for Prophet
        prophet_df = df.reset_index()[['datetime', 'Global_active_power']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train Prophet with cmdstanpy backend (silent mode)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, interval_width=0.95)
        model.fit(prophet_df)
        
        # Forecast next 100 periods
        future = model.make_future_dataframe(periods=100, freq='H')
        forecast = model.predict(future)
        
        print("[OK] Prophet baseline trained successfully")
        return model, forecast
    except Exception as e:
        print(f"[ERROR] Prophet training failed: {type(e).__name__}: {str(e)[:80]}")
        raise


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


def create_comparison_visualization(lstm_forecast, prophet_forecast, output_path):
    """Create comparison plot: LSTM vs Prophet vs Actual"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: LSTM vs Prophet vs Actual (first 200 samples)
    axes[0].plot(lstm_forecast.index[:200], lstm_forecast['actual'][:200], label='Actual', color='blue', linewidth=2.5)
    axes[0].plot(lstm_forecast.index[:200], lstm_forecast['predicted'][:200], label='LSTM Prediction', color='red', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Global Active Power (Normalized)')
    axes[0].set_title('Model Comparison: LSTM vs Actual (First 200 samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    axes[1].plot(lstm_forecast.index, lstm_forecast['error'], label='LSTM Error', color='green', linewidth=1, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(lstm_forecast.index, lstm_forecast['error'], 0, alpha=0.2, color='green')
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title('LSTM Prediction Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved comparison visualization to {output_path}")


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
    
    print(f"[OK] Saved visualization to {output_path}")


def calculate_metrics(forecast_df):
    """Calculate performance metrics"""
    mae = mean_absolute_error(forecast_df['actual'], forecast_df['predicted'])
    rmse = np.sqrt(mean_squared_error(forecast_df['actual'], forecast_df['predicted']))
    mape = mean_absolute_percentage_error(forecast_df['actual'], forecast_df['predicted'])
    
    return {'mae': round(mae, 6), 'rmse': round(rmse, 6), 'mape': round(mape, 6)}


def main():
    print("=" * 60)
    print("[*] FORECASTING PIPELINE - ENHANCED VERSION")
    print("=" * 60)
    print(f"[*] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Python: {sys.version.split()[0]}")
    print(f"[*] PyTorch: {torch.__version__}")
    print("=" * 60)
    
    # Log to file
    with open(LOG_PATH, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Pipeline started: {datetime.now().isoformat()}\n")
        f.write(f"Python {sys.version}\n")
        f.write(f"PyTorch {torch.__version__}\n")
        f.write(f"{'='*60}\n")
    
    try:
        # Configure wandb with API key from environment
        wandb_enabled = False
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key and len(wandb_api_key) >= 40:
            try:
                wandb.login(key=wandb_api_key)
                wandb.init(project="energy-forecasting", mode="online")
                wandb_enabled = True
                print("[OK] Weights & Biases initialized with API key")
            except Exception as e:
                print(f"[WARN] Weights & Biases failed: {type(e).__name__}: {e}")
        else:
            print("[WARN] WANDB not configured, running in offline mode")
            try:
                wandb.init(mode="disabled")
            except:
                pass

        print("\n[*] Loading and preprocessing data...")
        df = preprocess()
        df_feat = create_features(df)
        print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Optuna hyperparameter optimization
        print("\n[*] Starting Hyperparameter Optimization with Optuna...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, df_feat), n_trials=5)  # Reduced from 10 to 5

        best_params = study.best_params
        if wandb_enabled:
            try:
                wandb.log(best_params)
            except:
                pass
        print(f"[OK] Best params found: {best_params}")

        # Walk-forward validation
        target = "Global_active_power"
        wf_results = walk_forward_validation(df_feat, best_params, n_folds=3)

        # Retrain best model on full data
        print("\n[*] Retraining final model on full data...")
        target = "Global_active_power"
        X_all, y_all = create_sequences(df_feat, target, best_params["window"])
        best_model = train_lstm(X_all, y_all, best_params)

        save_best_outputs(study, best_model)

        # Generate LSTM forecasts
        print("\n[*] Generating LSTM Forecasts...")
        lstm_forecast_df = generate_forecasts(best_model, df_feat, target, best_params["window"])
        
        # Generate Prophet baseline
        print("\n[*] Generating Prophet Baseline...")
        prophet_model, prophet_forecast = train_prophet_baseline(df_feat)
        
        # Save forecasts to CSV
        forecast_csv_path = RESULT_PATH / "forecasts.csv"
        lstm_forecast_df.to_csv(forecast_csv_path)
        print(f"[OK] Saved LSTM forecasts to {forecast_csv_path}")
        
        # Create visualizations
        viz_path = RESULT_PATH / "forecast_visualization.png"
        create_visualization(lstm_forecast_df, viz_path)
        
        comp_viz_path = RESULT_PATH / "model_comparison.png"
        create_comparison_visualization(lstm_forecast_df, prophet_forecast, comp_viz_path)
        
        # Calculate metrics
        lstm_metrics = calculate_metrics(lstm_forecast_df)
        
        # Save monitoring data
        monitoring_file = MONITORING_PATH / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'lstm_metrics': lstm_metrics,
            'walk_forward_results': wf_results,
            'best_params': best_params,
            "optimization_trials": 5
        }
        
        with open(monitoring_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2)
        
        # Log file
        log_content = f"""
TRAINING COMPLETED: {datetime.now()}
{'=' * 60}

LSTM MODEL METRICS:
  MAE:  {lstm_metrics['mae']:.6f}
  RMSE: {lstm_metrics['rmse']:.6f}
  MAPE: {lstm_metrics['mape']:.6f}

WALK-FORWARD VALIDATION RESULTS:
  Average MAE:  {wf_results['average_mae']:.6f}
  Average RMSE: {wf_results['average_rmse']:.6f}
  Average MAPE: {wf_results['average_mape']:.6f}

BEST HYPERPARAMETERS:
{json.dumps(best_params, indent=2)}

FILES GENERATED:
  - forecasts.csv
  - forecast_visualization.png
  - model_comparison.png
  - best_params.json
  - best_model.pt
  - metrics.json
"""

        with open(LOG_PATH, "w", encoding='utf-8') as f:
            f.write(log_content)

        # Save final metrics
        metrics = {
            "lstm_model": lstm_metrics,
            "walk_forward_validation": {
                "average_mae": wf_results['average_mae'],
                "average_rmse": wf_results['average_rmse'],
                "average_mape": wf_results['average_mape'],
                "num_folds": 3
            },
            "best_params": best_params,
            "optimization_trials": 5,
            "timestamp": datetime.now().isoformat()
        }

        with open(RESULT_PATH / "metrics.json", "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("[OK] TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"[*] Final Metrics:")
        print(f"  LSTM MAE: {lstm_metrics['mae']:.6f}")
        print(f"  LSTM RMSE: {lstm_metrics['rmse']:.6f}")
        print(f"  LSTM MAPE: {lstm_metrics['mape']:.6f}")
        print(f"  Walk-Forward Avg RMSE: {wf_results['average_rmse']:.6f}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"[ERROR] FILE ERROR: {e}")
        print("   Make sure data files exist in ./data/raw/")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] VALUE ERROR: {e}")
        print("   Check your data format or parameters")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] RUNTIME ERROR: {e}")
        print("   Check GPU/CUDA availability or memory")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] UNEXPECTED ERROR: {type(e).__name__}")
        print(f"   {e}")
        print("   Check logs/training.log for details")
        sys.exit(1)


if __name__ == "__main__":
    try:
        print("[*] Starting Energy Forecasting Pipeline...")
        main()
        print("[OK] Pipeline completed successfully!")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"[ERROR] FILE ERROR: {e}")
        print("   Make sure data files exist in ./data/raw/")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] VALUE ERROR: {e}")
        print("   Check your data format or parameters")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] RUNTIME ERROR: {e}")
        print("   Check GPU/CUDA availability or memory")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] UNEXPECTED ERROR: {type(e).__name__}")
        print(f"   {e}")
        print("   Check logs/training.log for details")
        sys.exit(1)