import pandas as pd


def get_features_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "target",
            "date",
            "timestamp",
            "ticker",
            "direction",
        ]
    )


def get_features_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "direction",
            "date",
            "timestamp",
            "ticker",
            "target",
        ]
    )
