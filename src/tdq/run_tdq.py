import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tdq.tdq_checks import (
    schema_check,
    null_check,
    duplicate_check,
    freshness_check,
    datatype_validity
)

from src.cleaning.cleaning_functions import (
    clean_visitor_events,
    clean_applications,
    clean_accounts,
    clean_transactions
)

# -----------------------------
# Load datasets
# -----------------------------
DATA_PATH = "data/"

datasets = {
    "visitor_events": pd.read_csv(f"/Users/gritv/Desktop/projects /data/visitor_events.csv"),
    "applications": pd.read_csv(f"/Users/gritv/Desktop/projects /data/applications.csv"),
    "accounts": pd.read_csv(f"/Users/gritv/Desktop/projects /data/accounts.csv"),
    "transactions": pd.read_csv(f"/Users/gritv/Desktop/projects /data/transactions.csv"),
    "marketing_source": pd.read_csv(f"/Users/gritv/Desktop/projects /data/marketing_source.csv")
}

# Expected schema
expected_schema = {
    "visitor_events": {
        "event_id": "object",
        "visitor_id": "int64",
        "event_type": "object",
        "device_type": "object",
        "marketing_source": "object",
        "event_timestamp": "object",
        "session_id": "object",
        "geo_country": "object",
        "geo_city": "object"
    },
    "applications": {
        "application_id": "object",
        "visitor_id": "int64",
        "application_date": "object",
        "status": "object",
        "credit_score": "float64",
        "income": "int64",
        "loan_amount": "int64",
        "product_type": "object",
        "source_channel": "object"
    },
    "accounts": {
        "account_id": "object",
        "application_id": "object",
        "account_open_date": "object",
        "account_type": "object",
        "initial_deposit": "int64",
        "kyc_status": "object"
    },
    "transactions": {
        "transaction_id": "object",
        "account_id": "object",
        "transaction_timestamp": "object",
        "amount": "float64",
        "transaction_type": "object",
        "merchant_category": "object",
        "channel": "object"
    },
    "marketing_source": {
        "source": "object",
        "channel_cost": "float64",
        "target_demographic": "object"
    }
}

# -----------------------------
# TDQ RUNNER FUNCTION
# -----------------------------
def run_tdq(datasets):

    tdq_results = []

    for name, df in datasets.items():

        schema_miss, schema_extra, dtype_mismatch = schema_check(df, expected_schema[name])
        nulls = null_check(df)
        dups = duplicate_check(df)

        ts_cols = [c for c in df.columns if "timestamp" in c or "date" in c]

        if len(ts_cols) > 0:
            fresh = freshness_check(df, ts_cols[0])
        else:
            fresh = {"missing_ts": None, "time_gaps": None}

        numeric_cols = [
            c for c, t in expected_schema[name].items()
            if ("int" in t or "float" in t)
        ]
        type_issues = datatype_validity(df, numeric_cols)

        tdq_results.append({
            "table": name,
            "missing_columns": len(schema_miss),
            "extra_columns": len(schema_extra),
            "dtype_mismatch": len(dtype_mismatch),
            "null_issues": (nulls > 0).sum(),
            "duplicate_rows": dups,
            "missing_timestamps": fresh.get("missing_ts"),
            "time_gaps": fresh.get("time_gaps")
        })

    return pd.DataFrame(tdq_results)


# -----------------------------
# EXECUTION
# -----------------------------

# TDQ BEFORE CLEANING
tdq_raw = run_tdq(datasets)
tdq_raw.to_csv(f"{DATA_PATH}/tdq_raw.csv", index=False)
print("✓ Saved TDQ raw report")

# CLEAN DATA
cleaned_data = {
    "visitor_events": clean_visitor_events(datasets["visitor_events"]),
    "applications": clean_applications(datasets["applications"]),
    "accounts": clean_accounts(datasets["accounts"]),
    "transactions": clean_transactions(datasets["transactions"]),
    "marketing_source": datasets["marketing_source"]
}

# TDQ AFTER CLEANING
tdq_clean = run_tdq(cleaned_data)
tdq_clean.to_csv(f"{DATA_PATH}/tdq_clean.csv", index=False)
print("✓ Saved TDQ cleaned report")
