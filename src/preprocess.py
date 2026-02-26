import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path(os.getenv("DATASET_PATH", "data/raw/household_power_consumption.zip"))
OUT_PATH = Path("data/processed/processed.csv")


def preprocess():
    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        compression="zip",
        na_values=["?"],
        low_memory=False,
    )

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")

    # hourly resample
    df = df.resample("h").mean()

    df = df.ffill()

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)

    return df


if __name__ == "__main__":
    preprocess()