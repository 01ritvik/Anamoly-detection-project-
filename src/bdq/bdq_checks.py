import pandas as pd

def check_missing_visitors(applications, visitors):
    """Applications referencing visitor_ids that do not exist."""
    missing = ~applications["visitor_id"].isin(visitors["visitor_id"])
    return missing.sum()

def check_missing_applications(accounts, applications):
    """Accounts referencing application_ids that do not exist."""
    missing = ~accounts["application_id"].isin(applications["application_id"])
    return missing.sum()

def check_missing_accounts(transactions, accounts):
    """Transactions referencing account_ids that do not exist."""
    missing = ~transactions["account_id"].isin(accounts["account_id"])
    return missing.sum()


def check_credit_score(applications):
    """Credit score must be between 300 and 850."""
    invalid = (applications["credit_score"] < 300) | (applications["credit_score"] > 850)
    return invalid.sum()

def check_negative_income(applications):
    """Income cannot be negative."""
    return (applications["income"] < 0).sum()

def check_invalid_status(applications):
    """Application status must be: submitted/approved/rejected."""
    valid = ["submitted", "approved", "rejected"]
    return (~applications["status"].isin(valid)).sum()

def check_negative_deposit(accounts):
    """Initial deposit cannot be negative."""
    return (accounts["initial_deposit"] < 0).sum()

def check_invalid_kyc(accounts):
    """KYC must be pending/verified/failed."""
    valid = ["verified", "pending", "failed"]
    return (~accounts["kyc_status"].isin(valid)).sum()

def check_invalid_tx_amount(transactions):
    """Transaction amount must be > 0."""
    return (transactions["amount"] <= 0).sum()

def check_marketing_sources(visitors, marketing):
    """Marketing source must match metadata table."""
    valid = marketing["source"].unique()
    return (~visitors["marketing_source"].isin(valid)).sum()
