import matplotlib.pyplot as plt
import os

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

def ensure_folder():
    os.makedirs(VISUALS_DIR, exist_ok=True)


def save_distribution_plot(rx):
    ensure_folder()
    plt.figure(figsize=(10, 4))
    plt.hist(rx["final_anomaly_score"], bins=50)
    plt.title("Distribution of Final Anomaly Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "distribution.png"))
    plt.close()


def save_daily_anomaly_plot(rx):
    ensure_folder()
    rx["date"] = rx["transaction_timestamp"].dt.date
    daily_anoms = rx.groupby("date")["final_anomaly_flag"].sum().reset_index()

    plt.figure(figsize=(12, 4))
    plt.plot(daily_anoms["date"], daily_anoms["final_anomaly_flag"], marker="o")
    plt.title("Daily Count of Final Anomalous Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "daily_anomalies.png"))
    plt.close()


def save_ts_anomaly_plot(tx_hour):
    ensure_folder()
    plt.figure(figsize=(14, 4))
    plt.plot(tx_hour["transaction_timestamp"], tx_hour["tx_amount"],
             label="Hourly Amount", linewidth=0.7)

    anom = tx_hour[tx_hour["ts_anomaly"]]
    plt.scatter(anom["transaction_timestamp"], anom["tx_amount"],
                color="red", s=30, label="TS Anomaly")

    plt.title("Hourly Transaction Amount with Detected Time-Series Anomalies")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "ts_anomalies.png"))
    plt.close()
