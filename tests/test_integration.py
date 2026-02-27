"""
Integration tests for the forecasting pipeline
Tests data pipeline, model training, and evaluation end-to-end
"""

import sys
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import preprocess
from feature_engineering import create_features
from models import LSTMModel
from train import (
    create_sequences, 
    train_lstm, 
    walk_forward_validation,
    train_prophet_baseline,
    calculate_metrics,
    EarlyStopping,
    PerformanceMonitor
)


class TestDataPipeline:
    """Test data preprocessing and feature engineering"""
    
    def test_preprocess_output_shape(self):
        """Test that preprocessing returns a valid DataFrame"""
        df = preprocess()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Global_active_power' in df.columns
    
    def test_preprocess_no_missing_values(self):
        """Test that preprocessing handles missing values"""
        df = preprocess()
        assert df.isnull().sum().sum() == 0, "Preprocessed data should have no missing values"
    
    def test_feature_engineering_output(self):
        """Test that feature engineering produces additional features"""
        df = preprocess()
        df_feat = create_features(df)
        
        assert isinstance(df_feat, pd.DataFrame)
        assert len(df_feat) > 0
        # Check for engineered features
        assert 'hour' in df_feat.columns
        assert 'dayofweek' in df_feat.columns
        assert 'month' in df_feat.columns
        assert 'Global_active_power_lag_1' in df_feat.columns
    
    def test_feature_engineering_length(self):
        """Test that feature engineering doesn't duplicate data incorrectly"""
        df = preprocess()
        df_feat = create_features(df)
        # After feature engineering and dropna(), should have fewer rows
        assert len(df_feat) < len(df)
        assert len(df_feat) > 0


class TestSequenceCreation:
    """Test sequence creation for LSTM"""
    
    def test_create_sequences_output_shape(self):
        """Test that sequences have correct shape"""
        # Create dummy data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        X, y = create_sequences(data, 'target', window=24)
        
        assert X.shape[0] == y.shape[0], "X and y should have same first dimension"
        assert X.shape[1] == 24, "Window size should match expected length"
        assert X.shape[0] == len(data) - 24, "Sequence count should be data_length - window"
    
    def test_create_sequences_no_empty(self):
        """Test that sequences are created correctly"""
        data = pd.DataFrame({
            'feature': np.arange(100),
            'target': np.arange(100, 200)
        })
        
        X, y = create_sequences(data, 'target', window=10)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[1] == 10


class TestLSTMModel:
    """Test LSTM model functionality"""
    
    def test_lstm_model_creation(self):
        """Test that LSTM model can be created"""
        model = LSTMModel(input_size=5, hidden_size=32, num_layers=2, dropout=0.2)
        assert isinstance(model, torch.nn.Module)
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass produces correct output shape"""
        model = LSTMModel(input_size=5, hidden_size=32, num_layers=1, dropout=0.0)
        
        # Create dummy input: (batch_size, seq_len, input_size)
        X = torch.randn(10, 24, 5)
        output = model(X)
        
        assert output.shape[0] == 10, "Batch size should be preserved"
        assert output.shape[1] == 1, "Output should be single value per sample"
    
    def test_lstm_training_reduces_loss(self):
        """Test that LSTM training reduces loss"""
        model = LSTMModel(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        X = torch.randn(20, 24, 3)
        y = torch.randn(20)
        
        initial_loss = None
        for epoch in range(10):
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        assert final_loss < initial_loss, "Loss should decrease during training"


class TestEarlyStoppingAndMonitoring:
    """Test early stopping and performance monitoring"""
    
    def test_early_stopping_initialization(self):
        """Test EarlyStopping can be created"""
        es = EarlyStopping(patience=3, min_delta=0.001)
        assert es.patience == 3
        assert es.early_stop == False
    
    def test_early_stopping_activation(self):
        """Test EarlyStopping activates after patience epochs"""
        es = EarlyStopping(patience=2, min_delta=0.001)
        
        # Simulate non-improving loss
        es(1.0)
        assert es.early_stop == False
        es(1.0)
        assert es.early_stop == False
        es(1.0)
        assert es.early_stop == True
    
    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor()
        assert 'train_loss' in monitor.history
        assert 'val_loss' in monitor.history
    
    def test_performance_monitor_update(self):
        """Test PerformanceMonitor update"""
        monitor = PerformanceMonitor()
        monitor.update(0, 1.5, 1.4, 0.1, 0.2)
        
        assert len(monitor.history['epoch']) == 1
        assert monitor.history['train_loss'][0] == 1.5
        assert monitor.history['val_mae'][0] == 0.1


class TestWalkForwardValidation:
    """Test walk-forward validation"""
    
    def test_walk_forward_returns_results(self):
        """Test walk_forward_validation returns proper structure"""
        df = preprocess()
        df_feat = create_features(df)
        
        params = {
            'window': 24,
            'hidden': 32,
            'layers': 1,
            'dropout': 0.2,
            'lr': 0.001
        }
        
        # Use small fold size for testing
        results = walk_forward_validation(df_feat, params, n_folds=2)
        
        assert 'fold_results' in results
        assert 'average_mae' in results
        assert 'average_rmse' in results
        assert len(results['fold_results']) == 2
    
    def test_walk_forward_metrics_valid(self):
        """Test that walk_forward metrics are valid numbers"""
        df = preprocess()
        df_feat = create_features(df)
        
        params = {
            'window': 24,
            'hidden': 32,
            'layers': 1,
            'dropout': 0.2,
            'lr': 0.001
        }
        
        results = walk_forward_validation(df_feat, params, n_folds=2)
        
        assert results['average_mae'] > 0
        assert results['average_rmse'] > 0
        assert results['average_mape'] >= 0


class TestProphetBaseline:
    """Test Prophet baseline model"""
    
    def test_prophet_training(self):
        """Test Prophet model can be trained"""
        df = preprocess()
        df_feat = create_features(df)
        
        model, forecast = train_prophet_baseline(df_feat)
        
        assert model is not None
        assert forecast is not None
        assert 'yhat' in forecast.columns


class TestMetricsCalculation:
    """Test metrics calculation"""
    
    def test_calculate_metrics(self):
        """Test metrics are calculated correctly"""
        forecast_df = pd.DataFrame({
            'actual': [1.0, 2.0, 3.0, 4.0, 5.0],
            'predicted': [1.1, 2.1, 2.9, 4.1, 4.9],
            'error': [-0.1, -0.1, 0.1, -0.1, 0.1]
        })
        
        metrics = calculate_metrics(forecast_df)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mape'] > 0


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_pipeline_components(self):
        """Test that all pipeline components work together"""
        # Data preprocessing
        df = preprocess()
        assert len(df) > 0
        
        # Feature engineering
        df_feat = create_features(df)
        assert len(df_feat) > 0
        
        # Sequence creation
        X, y = create_sequences(df_feat, 'Global_active_power', 24)
        assert len(X) > 0
        
        # Model training
        params = {
            'window': 24,
            'hidden': 16,
            'layers': 1,
            'dropout': 0.1,
            'lr': 0.001
        }
        
        split = int(len(X) * 0.8)
        model = train_lstm(X[:split], y[:split], params)
        assert isinstance(model, LSTMModel)
        
        # Prediction
        with torch.no_grad():
            predictions = model(torch.tensor(X[split:], dtype=torch.float32))
        assert predictions.shape[0] == len(X[split:])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
