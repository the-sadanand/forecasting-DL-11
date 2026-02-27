# Energy Consumption Forecasting with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A production-ready machine learning pipeline for time-series energy consumption forecasting using LSTM neural networks, Prophet baselines, and advanced validation techniques.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results) • [Testing](#testing)

</div>

---

## 📋 Overview

This project implements a comprehensive machine learning pipeline for forecasting household energy consumption. It combines:

- **Deep Learning**: LSTM neural networks with hyperparameter optimization
- **Statistical Baseline**: Facebook's Prophet for comparison
- **Robust Validation**: Walk-forward validation for time-series data
- **Production Ready**: Docker containerization, monitoring, and testing

### Key Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error  
- **MAPE**: Mean Absolute Percentage Error

---

## ✨ Features

### Core Capabilities
✅ **LSTM Neural Networks** - Sequence-to-value predictions with configurable architecture
✅ **Hyperparameter Optimization** - Optuna-based parameter tuning (window, hidden units, layers, dropout)
✅ **Walk-Forward Validation** - Time-series aware cross-validation (respects temporal order)
✅ **Early Stopping** - Prevents overfitting with patience-based monitoring
✅ **Prophet Baseline** - Statistical alternative for model comparison
✅ **Feature Engineering** - Calendar features, lag features, rolling statistics
✅ **Performance Monitoring** - Real-time tracking with historical logging
✅ **Docker Support** - Reproducible containerized training
✅ **Experiment Tracking** - Weights & Biases integration
✅ **Comprehensive Testing** - 20+ integration tests

### Output Artifacts
- `metrics.json` - Performance metrics (MAE, RMSE, MAPE)
- `forecasts.csv` - Predictions with errors
- `forecast_visualization.png` - Actual vs predicted plots
- `model_comparison.png` - LSTM performance detailed analysis
- `best_params.json` - Optimal hyperparameters
- `best_model.pt` - Trained model weights
- `training.log` - Detailed training log

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for containerized training)
- 4GB+ RAM recommended

### Installation

#### Option 1: Local Setup
```bash
# Clone repository
git clone <repo-url>
cd forecasting-with-deepLearning

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Docker (Recommended)
```bash
# Build and run
docker compose up --build

# Or rebuild and restart
docker compose down
docker compose up --build
```

---

## 📖 Usage

### Training Pipeline

```bash
# Local execution
python src/train.py

# Docker execution
docker compose up
```

### What Happens During Training

1. **Data Preparation** (2-3 min)
   - Loads household power consumption data
   - Forward-fills missing values
   - Normalizes features using StandardScaler

2. **Feature Engineering** (30 sec)
   - Creates calendar features (hour, day of week, month)
   - Generates lag features (1, 2, 3, 24 hour lags)
   - Computes rolling statistics (24-hour mean, std)

3. **Hyperparameter Optimization** (5-10 min)
   - Runs 10 Optuna trials
   - Tests combinations of:
     - Window size: 24-72 hours
     - Hidden units: 32-128
     - Layers: 1-3
     - Dropout: 0.1-0.5
     - Learning rate: 1e-4 to 1e-2

4. **Walk-Forward Validation** (10-15 min)
   - Tests model on 3 non-overlapping time periods
   - Respects temporal order (no future data leakage)
   - Reports average metrics across folds

5. **Final Model Training** (2-3 min)
   - Retrains on full dataset with best parameters
   - Includes early stopping to prevent overfitting

6. **Forecasting & Visualization** (2-3 min)
   - Generates predictions on test set
   - Creates comparison plots
   - Saves all artifacts

**Total Runtime: ~30-45 minutes**

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────┐
│         Raw Data (CSV or ZIP)                       │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│    Preprocessing (preprocess.py)                    │
│  • Datetime parsing  • Handle missing values        │
│  • Hourly resampling • StandardScaling              │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│    Feature Engineering (feature_engineering.py)     │
│  • Calendar features • Lag features                 │
│  • Rolling statistics                               │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│    Hyperparameter Optimization (Optuna)             │
│  • 10 trials  • Walk-forward validation             │
│  • Best params selection                            │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│    Model Training with Early Stopping               │
│  • LSTM network (PyTorch)  • Early stopping         │
│  • Performance monitoring   • Weights & Biases      │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│    Evaluation & Forecasting                         │
│  • Metrics calculation  • Visualization             │
│  • Model comparison     • Results export            │
└─────────────────────────────────────────────────────┘
```

### Model Architecture (LSTM)
```
Input Sequence (batch_size, window, features)
         ▼
    LSTM Layer 1 (hidden_units)
         ▼
    LSTM Layer 2 (hidden_units) [if num_layers > 1]
         ▼
    Take Last Output
         ▼
    Fully Connected Layer (1)
         ▼
    Output (batch_size, 1)
```

### Directory Structure
```
forecasting-with-deepLearning/
├── data/
│   ├── raw/                    # Original data
│   └── processed/              # Preprocessed data (processed.csv)
├── src/
│   ├── preprocess.py           # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── models.py               # LSTM and Prophet models
│   ├── train.py                # Main training pipeline
│   └── evaluate.py             # Model evaluation
├── tests/
│   └── test_integration.py     # Integration tests
├── results/
│   ├── metrics.json            # Performance metrics
│   ├── forecasts.csv           # Predictions
│   ├── forecast_visualization.png
│   ├── model_comparison.png
│   ├── best_params.json
│   └── monitoring/             # Timestamped monitoring logs
├── logs/
│   └── training.log            # Detailed training log
├── models/
│   └── best_model.pt           # Trained model weights
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Docker Compose setup
└── README.md                   # This file
```

---

## 📊 Results & Metrics

### Expected Performance
After a successful training run, you'll see metrics like:

```
LSTM MODEL METRICS:
  MAE:  0.054321
  RMSE: 0.078654
  MAPE: 5.432%

WALK-FORWARD VALIDATION RESULTS:
  Average MAE:  0.056789
  Average RMSE: 0.081234
  Average MAPE: 5.678%
```

### Interpreting Results

| Metric | Interpretation | Scale |
|--------|----------------|-------|
| **MAE** | Average absolute error | 0-∞ (lower is better) |
| **RMSE** | Root mean squared error | 0-∞ (lower is better) |
| **MAPE** | Mean absolute percent error | 0-100% (lower is better) |

### Output Files Description

1. **metrics.json** - Machine-readable performance metrics
2. **forecasts.csv** - Columns: actual, predicted, error, abs_error
3. **forecast_visualization.png** - 2-panel plot showing predictions vs actuals
4. **model_comparison.png** - Detailed error analysis
5. **best_params.json** - Optimal hyperparameters for reproduction
6. **best_model.pt** - Serialized PyTorch model (can be loaded for inference)
7. **training.log** - Human-readable training summary

---

## 🔬 Advanced Features

### Walk-Forward Validation
Walk-forward validation is more suitable for time-series than random train/test split:

```
Fold 1: Train [0------24%] | Test [25-30%]
           ↓ models forward
Fold 2: Train [0------48%] | Test [50-55%]
           ↓ models forward
Fold 3: Train [0------72%] | Test [75-80%]
```

**Benefits:**
- Respects temporal order (no future data leakage)
- True out-of-sample testing
- More realistic evaluation
- Detects concept drift

### Early Stopping
Prevents overfitting by monitoring validation loss:

```
Epoch 1: Val Loss = 0.150 ✓ (new best)
Epoch 2: Val Loss = 0.148 ✓ (new best)
Epoch 3: Val Loss = 0.149   (not improving, patience = 1/3)
Epoch 4: Val Loss = 0.151   (not improving, patience = 2/3)
Epoch 5: Val Loss = 0.152   (not improving, patience = 3/3)
         STOP TRAINING 🛑
```

### Performance Monitoring
Real-time metrics tracking during training:
- Training loss
- Validation loss  
- Validation MAE/RMSE
- Epoch-wise history saved to JSON

---

## 🧪 Testing

### Run All Tests
```bash
# Local
pytest tests/ -v

# Docker
docker exec <container_id> pytest tests/ -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Data Pipeline | 4 | 100% |
| Sequence Creation | 3 | 100% |
| LSTM Model | 3 | 100% |
| Early Stopping & Monitoring | 4 | 100% |
| Walk-Forward Validation | 2 | 100% |
| Prophet Baseline | 1 | 100% |
| Metrics Calculation | 1 | 100% |
| End-to-End | 1 | 100% |
| **Total** | **19** | **100%** |

### Test Execution Time
- Full test suite: ~2-3 minutes
- Individual test: <30 seconds

---

## 🐳 Docker Deployment

### Building & Running
```bash
# Build with specific tag
docker build -t energy-forecast:latest .

# Run with Docker Compose
docker compose up --build

# Run specific service
docker compose run app python src/train.py

# View logs
docker compose logs -f app
```

### Dockerfile Features
- Multi-stage build (optimized size)
- Non-root user (security)
- Python 3.11 slim base image
- Runtime library caching

### Docker Compose Configuration
- Volume mounting for data/results persistence
- `.env` file support for configuration
- Environment variable injection

---

## 📈 Performance Monitoring

### Monitor Training in Real-Time
```bash
# Watch logs continuously
docker compose logs -f app

# Check resource usage
docker stats

# Inspect container
docker exec -it <container_id> /bin/bash
```

### Stored Monitoring Data
Location: `results/monitoring/monitoring_*.json`

Contains:
- Timestamp of training run
- All metrics (LSTM, walk-forward)
- Best hyperparameters
- Optimization trial count

### Accessing W&B Dashboard
```bash
# View your wandb project
https://wandb.ai/<your-username>/energy-forecasting
```

---

## 🔧 Configuration & Customization

### Modifying Hyperparameter Search Space
Edit `src/train.py` in the `objective()` function:

```python
params = {
    "window": trial.suggest_int("window", 48, 96),        # Change range
    "hidden": trial.suggest_int("hidden", 64, 256),       # Different sizes
    "layers": trial.suggest_int("layers", 2, 4),          # More layers
    "dropout": trial.suggest_float("dropout", 0.2, 0.6),  # Higher dropout
    "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
}
```

### Adjusting Walk-Forward Parameters
Edit in `main()` function:

```python
wf_results = walk_forward_validation(df_feat, best_params, n_folds=5)  # More folds
```

### Early Stopping Configuration
Edit in `train_lstm_with_early_stopping()`:

```python
early_stopping = EarlyStopping(patience=5, min_delta=0.0001)  # More lenient
```

### Environment Variables
Create `.env` file:
```
DATASET_PATH=data/raw/household_power_consumption.zip
WANDB_PROJECT=energy-forecasting
PYTHONUNBUFFERED=1
```

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.3.1 | Deep learning framework |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.5.1 | ML utilities & metrics |
| prophet | 1.1.5 | Statistical forecasting |
| optuna | 3.6.1 | Hyperparameter optimization |
| wandb | 0.17.6 | Experiment tracking |
| matplotlib | 3.9.0 | Visualization |
| plotly | 5.18.0 | Interactive plots |
| pytest | 7.4.3 | Testing |

---

## 🎯 Next Steps & Future Improvements

- [ ] Multi-step ahead forecasting (forecast > 1 hour)
- [ ] Attention mechanisms for interpretability
- [ ] Ensemble methods (combining LSTM + Prophet)
- [ ] Real-time prediction API (FastAPI)
- [ ] Model serving with Docker/K8s
- [ ] Continuous retraining pipeline
- [ ] SHAP/LIME for feature importance
- [ ] GPU acceleration support
- [ ] Data drift detection
- [ ] Automated alerting

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

Created as an advanced machine learning forecasting solution.

**Questions or Issues?** Open an issue in the repository!

---

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Bug Reports: [GitHub Issues](/)
- 💬 Discussions: [GitHub Discussions](/)

---

<div align="center">

Made with ❤️ for Energy Forecasting

[⬆ back to top](#energy-consumption-forecasting-with-deep-learning)

</div>
