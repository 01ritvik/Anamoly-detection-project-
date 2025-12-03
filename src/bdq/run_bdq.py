import pandas as pd
from src.bdq.bdq_checks import (
    check_missing_visitors,
    check_missing_applications,
    check_missing_accounts,
    check_credit_score,
    check_negative_income,
    check_invalid_status,
    check_negative_deposit,
    check_invalid_kyc,
    check_invalid_tx_amount,
    check_marketing_sources
)

# -------------------------
# Load CLEANED data
# -------------------------
visitor = pd.read_csv("data/cleaned_visitor_events.csv")
applications = pd.read_csv("data/cleaned_applications.csv")
accounts = pd.read_csv("data/cleaned_accounts.csv")
transactions = pd.read_csv("data/cleaned_transactions.csv")
marketing = pd.read_csv("data/marketing_source.csv")

# -------------------------
# Run BDQ Checks
# -------------------------
bdq_results = []

bdq_results.append({
    "check": "apps_missing_visitor",
    "num_issues": check_missing_visitors(applications, visitor)
})

bdq_results.append({
    "check": "accs_missing_application",
    "num_issues": check_missing_applications(accounts, applications)
})

bdq_results.append({
    "check": "tx_missing_account",
    "num_issues": check_missing_accounts(transactions, accounts)
})

bdq_results.append({
    "check": "bad_credit_score",
    "num_issues": check_credit_score(applications)
})

bdq_results.append({
    "check": "negative_income",
    "num_issues": check_negative_income(applications)
})

bdq_results.append({
    "check": "invalid_status",
    "num_issues": check_invalid_status(applications)
})

bdq_results.append({
    "check": "negative_deposit",
    "num_issues": check_negative_deposit(accounts)
})

bdq_results.append({
    "check": "bad_kyc_status",
    "num_issues": check_invalid_kyc(accounts)
})

bdq_results.append({
    "check": "invalid_tx_amount",
    "num_issues": check_invalid_tx_amount(transactions)
})

bdq_results.append({
    "check": "marketing_invalid_cost",
    "num_issues": check_marketing_sources(visitor, marketing)
})

# -------------------------
# Convert to DataFrame
# -------------------------
bdq_df = pd.DataFrame(bdq_results)
print(bdq_df)

# -------------------------
# Save report
# -------------------------
bdq_df.to_csv("data/bdq_report.csv", index=False)
print("\n✓ BDQ report saved to data/bdq_report.csv")
