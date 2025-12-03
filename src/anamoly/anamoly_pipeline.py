import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from scipy import stats
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.anamoly.visuals import (
    save_distribution_plot,
    save_daily_anomaly_plot,
    save_ts_anomaly_plot
)


plt.rcParams["figure.dpi"] = 120
np.random.seed(42)

from visuals import save_distribution_plot, save_daily_anomaly_plot, save_ts_anomaly_plot


def load_data(DATA_DIR="data"):
    tx_path = os.path.join(DATA_DIR, "cleaned_transactions.csv")
    df = pd.read_csv(tx_path, parse_dates=["transaction_timestamp"])
    print("transactions:", df.shape)
    return df


def preprocess(df):
    tx = df.copy()

    tx["transaction_timestamp"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
    tx = tx.sort_values("transaction_timestamp").reset_index(drop=True)

    tx["date"] = tx["transaction_timestamp"].dt.date
    tx["hour"] = tx["transaction_timestamp"].dt.hour
    tx["dayofweek"] = tx["transaction_timestamp"].dt.dayofweek

    tx["merchant_freq"] = tx.groupby("merchant_category")["merchant_category"].transform("count")
    tx["channel_freq"] = tx.groupby("channel")["channel"].transform("count")

    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0).astype(float)

    return tx


def build_hourly_series(tx):
    tx_hour = tx.set_index("transaction_timestamp").resample("1H").agg({
        "transaction_id": "count",
        "amount": "sum"
    }).rename(columns={"transaction_id": "tx_count", "amount": "tx_amount"}).fillna(0)

    return tx_hour.reset_index()


def detect_ts_anomalies(tx_hour, RESID_Z_THRESH=3.5, COUNT_Z_THRESH=4.0):
    """Detect time-series anomalies using STL residuals + count Z-scores."""
    series = tx_hour.set_index("transaction_timestamp")["tx_amount"]

    stl = STL(series, period=24, robust=True)
    res = stl.fit()

    tx_hour["stl_trend"] = res.trend.values
    tx_hour["stl_seasonal"] = res.seasonal.values
    tx_hour["stl_resid"] = res.resid.values

    tx_hour["stl_resid_z"] = stats.zscore(tx_hour["stl_resid"].fillna(0).values)
    tx_hour["tx_count_z"] = stats.zscore(tx_hour["tx_count"].values)

    tx_hour["ts_anom_resid"] = tx_hour["stl_resid_z"].abs() > RESID_Z_THRESH
    tx_hour["ts_anom_count"] = tx_hour["tx_count_z"].abs() > COUNT_Z_THRESH
    tx_hour["ts_anomaly"] = tx_hour["ts_anom_resid"] | tx_hour["ts_anom_count"]

    return tx_hour


def build_row_features(tx):
    rx = tx.copy()

    rx["amount_log"] = np.log1p(rx["amount"])
    rx["hour_sin"] = np.sin(2 * np.pi * rx["hour"] / 24)
    rx["hour_cos"] = np.cos(2 * np.pi * rx["hour"] / 24)

    rx["merchant_freq_norm"] = rx["merchant_freq"] / rx["merchant_freq"].max()
    rx["channel_freq_norm"] = rx["channel_freq"] / rx["channel_freq"].max()

    feature_cols = [
        "amount", "amount_log", "merchant_freq_norm",
        "channel_freq_norm", "hour_sin", "hour_cos"
    ]

    X = rx[feature_cols].fillna(0).values
    return rx, X


def detect_row_anomalies(rx, X):
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    lof_pred = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    def normalize_arr(a):
        a = np.array(a, dtype=float)
        if a.max() == a.min():
            return np.zeros_like(a)
        return (a - a.min()) / (a.max() - a.min())

    rx["iso_score"] = normalize_arr(iso_scores)
    rx["lof_score"] = normalize_arr(lof_scores)

    rx["iso_anomaly"] = rx["iso_score"] > 0.7
    rx["lof_anomaly"] = rx["lof_score"] > 0.7

    return rx


def combine_ts_and_row(rx, tx_hour, FINAL_SCORE_THRESH=0.65):
    tx_small = tx_hour[["transaction_timestamp", "ts_anomaly"]].copy()
    tx_small["hour_round"] = tx_small["transaction_timestamp"].dt.floor("H")

    rx["hour_round"] = rx["transaction_timestamp"].dt.floor("H")

    rx = rx.merge(
        tx_small[["hour_round", "ts_anomaly"]].drop_duplicates(),
        on="hour_round",
        how="left"
    )

    rx["ts_anomaly"] = rx["ts_anomaly"].fillna(False)

    rx["final_anomaly_score"] = 0.55 * rx["iso_score"] + \
                                0.45 * rx["lof_score"] + \
                                0.15 * rx["ts_anomaly"].astype(int)

    rx["final_anomaly_score"] = rx["final_anomaly_score"].clip(0, 1)
    rx["final_anomaly_flag"] = rx["final_anomaly_score"] > FINAL_SCORE_THRESH

    return rx


def save_outputs(rx, tx_hour, DATA_DIR="data"):
    out_folder = os.path.join(DATA_DIR, "reports", "anomaly")
    os.makedirs(out_folder, exist_ok=True)

    rx.to_csv(os.path.join(out_folder, "transaction_row_anomalies.csv"), index=False)
    tx_hour.to_csv(os.path.join(out_folder, "ts_hourly_anomalies.csv"), index=False)

    summary = {
        "total_transactions": len(rx),
        "row_level_anomalies": int(rx["final_anomaly_flag"].sum()),
        "hourly_anomalies": int(tx_hour["ts_anomaly"].sum())
    }

    print("✓ Saved transaction_row_anomalies.csv and ts_hourly_anomalies.csv to", out_folder)
    print("Summary:", summary)
    return summary


def run_pipeline(DATA_DIR="data"):
    tx = load_data(DATA_DIR)
    tx = preprocess(tx)

    tx_hour = build_hourly_series(tx)
    tx_hour = detect_ts_anomalies(tx_hour)

    rx, X = build_row_features(tx)
    rx = detect_row_anomalies(rx, X)

    rx = combine_ts_and_row(rx, tx_hour)

    summary = save_outputs(rx, tx_hour, DATA_DIR)
    print("📊 Saving visual plots to /visuals ...")
    save_distribution_plot(rx)
    save_daily_anomaly_plot(rx)
    save_ts_anomaly_plot(tx_hour)
    print("✓ Visuals saved in projects/visuals/")

    return rx, tx_hour, summary


if __name__ == "__main__":
    run_pipeline()
