import pandas as pd


def schema_check(df, expected_schema: dict):
    missing = set(expected_schema.keys()) - set(df.columns)
    extra = set(df.columns) - set(expected_schema.keys())
    dtype_mismatch = {}

    for col, expected_type in expected_schema.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                dtype_mismatch[col] = {"expected": expected_type, "actual": actual_type}

    return missing, extra, dtype_mismatch


def null_check(df):
    return (df.isnull().sum() / len(df) * 100).round(2)


def duplicate_check(df):
    return df.duplicated().sum()

def freshness_check(df, ts_col):
    missing_ts = df[ts_col].isna().sum()

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    df["_ts"] = ts

    max_ts = ts.max()

    df_sorted = df.sort_values("_ts")
    gaps = df_sorted["_ts"].diff().dt.total_seconds().fillna(0)
    time_gap_count = (gaps > 7200).sum()

    df.drop(columns=["_ts"], inplace=True)

    return {
        "max_timestamp": max_ts,
        "missing_ts": int(missing_ts),
        "time_gaps": int(time_gap_count)
    }


def datatype_validity(df, numeric_cols):
    issues = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        invalid = df[df[col].apply(lambda x: isinstance(x, str))]
        if len(invalid) > 0:
            issues[col] = f"{len(invalid)} invalid string values found"

    return issues
