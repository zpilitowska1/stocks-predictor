import pytest
import pandas as pd
from unittest.mock import patch
from stocks_predictor import main as main_module


# ========== Test: Invalid ticker ==========
def test_main_invalid_ticker(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "INVALID" if "ticker" in prompt else "2023-01-01",
    )
    main_module.main()
    captured = capsys.readouterr()
    assert "Invalid ticker symbol." in captured.out


# ========== Test: Invalid date format ==========
def test_main_invalid_date(monkeypatch, capsys):
    inputs = iter(["AAPL", "not-a-date"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    main_module.main()
    captured = capsys.readouterr()
    assert "Invalid date format" in captured.out


# ========== Test: Data loading error ==========
def test_main_data_loading_error(monkeypatch):
    inputs = iter(["AAPL", "2023-01-01"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with (
        patch(
            "stocks_predictor.main.load_and_prepare_data",
            side_effect=ValueError("file not found"),
        ),
        patch("stocks_predictor.main.print") as mock_print,
    ):
        main_module.main()
        mock_print.assert_any_call("Error loading data: file not found")


# ========== Test: No data found for ticker/date ==========
def test_main_no_data_found(monkeypatch):
    inputs = iter(["AAPL", "2023-01-01"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02"]),  # Different date
            "ticker": ["AAPL"],
            "target": [150],
            "feature1": [1],
            "feature2": [2],
        }
    )

    with (
        patch("stocks_predictor.main.load_and_prepare_data", return_value=df),
        patch("stocks_predictor.main.print") as mock_print,
    ):
        main_module.main()
        mock_print.assert_any_call("No data found for that date and ticker.")


# ========== Test: Successful prediction path ==========
def test_main_success(monkeypatch):
    inputs = iter(["AAPL", "2023-01-01"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    sample_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"]),
            "ticker": ["AAPL"],
            "target": [150.5],
            "feature1": [1.2],
            "feature2": [3.4],
        }
    )

    pred_df = pd.DataFrame({"target": [155.23], "direction": ["Up"]})

    with (
        patch("stocks_predictor.main.load_and_prepare_data", return_value=sample_df),
        patch("stocks_predictor.main.make_predictions", return_value=pred_df),
        patch("stocks_predictor.main.write_to_db") as mock_write,
        patch("stocks_predictor.main.print") as mock_print,
    ):
        main_module.main()
        mock_write.assert_called_once()
        mock_print.assert_any_call("Predictions saved to database.")
