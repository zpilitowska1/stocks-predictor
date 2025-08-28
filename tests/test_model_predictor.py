import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from stocks_predictor.model_predictor import make_predictions


@pytest.fixture
def example_df():
    data = {
        "ticker": ["AAPL", "AAPL"],
        "target": [150, 160],
        "direction": [1, 0],
        "date": pd.to_datetime(["2022-02-01", "2022-02-02"]),
        "timestamp": [1, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["ticker", "target", "direction", "date", "timestamp"])


@patch("stocks_predictor.model_predictor.load_model_and_scaler")
def test_make_predictions_basic(mock_load, example_df):
    mock_reg_model = MagicMock()
    mock_reg_model.predict.return_value = np.array([155, 165])
    mock_clf_model = MagicMock()
    mock_clf_model.predict.return_value = np.array([1, 1])
    mock_reg_scaler = MagicMock()
    mock_reg_scaler.transform.side_effect = lambda x: x
    mock_clf_scaler = MagicMock()
    mock_clf_scaler.transform.side_effect = lambda x: x
    mock_load.return_value = (
        mock_reg_model,
        mock_clf_model,
        mock_reg_scaler,
        mock_clf_scaler,
    )

    preds = make_predictions(example_df, "AAPL")
    assert (preds["target"] == [155, 165]).all()
    assert (preds["direction"] == [1, 1]).all()


@patch("stocks_predictor.model_predictor.load_model_and_scaler")
def test_make_predictions_empty_df(mock_load, empty_df):
    mock_reg_model = MagicMock()
    mock_reg_model.predict.return_value = np.array([])
    mock_clf_model = MagicMock()
    mock_clf_model.predict.return_value = np.array([])
    mock_reg_scaler = MagicMock()
    mock_reg_scaler.transform.side_effect = lambda x: x
    mock_clf_scaler = MagicMock()
    mock_clf_scaler.transform.side_effect = lambda x: x
    mock_load.return_value = (
        mock_reg_model,
        mock_clf_model,
        mock_reg_scaler,
        mock_clf_scaler,
    )

    preds = make_predictions(empty_df, "AAPL")
    assert preds.empty
