from stocks_predictor.data_loader import load_and_prepare_data
from stocks_predictor.model_predictor import make_predictions
from stocks_predictor.db_writer import write_to_db
import pandas as pd

TICKERS = ["AAPL", "QCOM", "GOOG"]


def main():
    # 1. Ask user input
    ticker = input(f"Enter ticker ({', '.join(TICKERS)}): ").strip().upper()
    if ticker not in TICKERS:
        print("Invalid ticker symbol.")
        return

    date_str = input("Enter date to predict (YYYY-MM-DD): ").strip()
    try:
        date = pd.to_datetime(date_str).date()
    except Exception as e:
        print(f"Invalid date format: {e}")
        return

    # 2. Load and filter data
    try:
        df = load_and_prepare_data("data\stock_data_fixed.csv")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    df = df[df["ticker"] == ticker]
    df = df[df["date"] == pd.to_datetime(date_str)]
    if df.empty:
        print("No data found for that date and ticker.")
        return

    # 3. Make predictions
    pred_df = make_predictions(df, ticker)
    regression_preds = pred_df["target"].round(2).tolist()
    direction_preds = pred_df["direction"].tolist()
    model_name = f"{ticker}_model"

    # 4. Write to DB
    write_to_db(
        predictions=regression_preds,
        directions=direction_preds,
        model_name=model_name,
        prediction_date=date_str,
        ticker=ticker,
    )

    print("Predictions saved to database.")


if __name__ == "__main__":
    main()
