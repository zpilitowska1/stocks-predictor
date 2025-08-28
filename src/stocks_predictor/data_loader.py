import pandas as pd


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except (ValueError, pd.errors.ParserError) as e:
        raise ValueError("Invalid date format in CSV") from e
    return df
