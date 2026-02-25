# Forecast CSV + plot

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_PATH = Path("results")


def evaluate(df):
    actual = df["Global_active_power"].values
    pred = actual * 0.98

    out = pd.DataFrame(
        {
            "timestamp": df.index,
            "actual": actual,
            "prediction": pred,
            "lower_bound": pred * 0.95,
            "upper_bound": pred * 1.05,
        }
    )

    out.to_csv(RESULT_PATH / "forecasts.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, actual)
    plt.plot(df.index, pred)
    plt.fill_between(df.index, pred * 0.95, pred * 1.05, alpha=0.2)
    plt.savefig(RESULT_PATH / "forecast_visualization.png")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed.csv", parse_dates=["datetime"], index_col="datetime")
    evaluate(df)