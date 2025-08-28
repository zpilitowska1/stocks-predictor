import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

db_params = {
    "host": "194.171.191.226",
    "port": "3432",
    "database": "group_17_warehouse",
    "user": "group_17_user",
    "password": "group_17_blockd25",
}

db_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"


def write_to_db(
    predictions: list[float | int],
    directions: list[int],
    model_name: str,
    prediction_date: str,
    ticker: str,
    table_name: str = "stock_predictions",
):
    """
    Writes regression and classification prediction results to a PostgreSQL table.

    Args:
        predictions (list): Regression predictions (floats).
        directions (list): Classification predictions (0 or 1).
        model_name (str): Model name.
        prediction_date (str): Prediction date from user input (YYYY-MM-DD).
        ticker (str): Stock ticker.
    """
    df = pd.DataFrame(
        {
            "model_name": model_name,
            "prediction_date": prediction_date,
            "ticker": ticker,
            "regression_prediction": predictions,
            "direction_prediction": directions,
        }
    )

    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="append", index=False)

    if not predictions or not directions:
        raise ValueError("Predictions and directions must not be empty.")
    if len(predictions) != len(directions):
        raise ValueError("Predictions and directions must have the same length.")
