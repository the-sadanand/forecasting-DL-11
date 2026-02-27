#!/bin/bash
set -e

echo "[*] Starting Energy Forecasting Pipeline in Docker"
echo "[*] Container: $(hostname)"
echo "[*] Python: $(python --version)"
echo "[*] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "============================================================"

# Pre-initialize Prophet/Stan backend to avoid runtime delays
echo "[*] Pre-initializing Prophet Stan backend..."
python -c "
import logging
logging.basicConfig(level=logging.WARNING)
try:
    from prophet import Prophet
    import pandas as pd
    # Quick test of Prophet to ensure Stan is ready
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100),
        'y': range(100)
    })
    print('[*] Testing Prophet...')
    m = Prophet(interval_width=0.95)
    m.fit(df)
    print('[OK] Prophet/Stan initialized successfully')
except Exception as e:
    print(f'[WARN] Prophet initialization: {e}')
" 2>&1 | grep -E "OK|WARN|ERROR|Testing" || true

echo ""
echo "============================================================"

# Run the training pipeline
python src/train.py

# Verify results
echo ""
echo "============================================================"
echo "[*] Verifying outputs..."
python verify_project.py

echo ""
echo "[OK] Container execution completed successfully!"
echo "[*] Check volumes for results in: results/, models/, logs/"

