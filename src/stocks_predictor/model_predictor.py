import joblib
import pandas as pd
from pathlib import Path
from stocks_predictor.feature_selector import (
    get_features_for_regression,
    get_features_for_classification,
)


def load_model_and_scaler(ticker: str):
    base = Path("models")
    reg_model = joblib.load(base / f"{ticker}_LR.pkl")
    clf_model = joblib.load(base / f"{ticker}_XGB.pkl")
    reg_scaler = joblib.load(base / f"{ticker}_SCALER_REG.save")
    clf_scaler = joblib.load(base / f"{ticker}_SCALER_CLF.save")
    return reg_model, clf_model, reg_scaler, clf_scaler


def make_predictions(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    reg_model, clf_model, reg_scaler, clf_scaler = load_model_and_scaler(ticker)
    ticker_df = df[df["ticker"] == ticker].copy()

    # Regression
    X_reg = get_features_for_regression(ticker_df)
    X_reg_scaled = reg_scaler.transform(X_reg)
    ticker_df["target"] = reg_model.predict(X_reg_scaled)

    # Classification
    X_clf = get_features_for_classification(ticker_df)
    X_clf_scaled = clf_scaler.transform(X_clf)
    ticker_df["direction"] = clf_model.predict(X_clf_scaled)

    return ticker_df
