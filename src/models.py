import torch
import torch.nn as nn
from prophet import Prophet
import pandas as pd

#  contains LSTM + Prophet baseline

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, df: pd.DataFrame):
        prophet_df = df.reset_index()[["datetime", "Global_active_power"]]
        prophet_df.columns = ["ds", "y"]
        self.model.fit(prophet_df)

    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods, freq="H")
        forecast = self.model.predict(future)
        return forecast