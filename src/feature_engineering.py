import pandas as pd


def create_features(df: pd.DataFrame, target="Global_active_power") -> pd.DataFrame:
    df_feat = df.copy()

    # calendar features
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month

    # lag features
    for lag in [1, 2, 3, 24]:
        df_feat[f"{target}_lag_{lag}"] = df_feat[target].shift(lag)

    # rolling
    df_feat[f"{target}_roll_mean_24"] = df_feat[target].rolling(24).mean()
    df_feat[f"{target}_roll_std_24"] = df_feat[target].rolling(24).std()

    df_feat = df_feat.dropna()

    return df_feat