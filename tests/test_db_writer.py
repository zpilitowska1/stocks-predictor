import pytest
from unittest.mock import patch, MagicMock
from stocks_predictor.db_writer import write_to_db


@patch("stocks_predictor.db_writer.create_engine")
def test_write_to_db_basic(mock_create_engine):
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    write_to_db(
        predictions=[1.1, 2.2],
        directions=[0, 1],
        model_name="model_test",
        prediction_date="2023-01-01",
        ticker="AAPL",
    )

    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        write_to_db(
            predictions=[1.1, 2.2],
            directions=[0, 1],
            model_name="model_test",
            prediction_date="2023-01-01",
            ticker="AAPL",
        )
        mock_to_sql.assert_called_once()


def test_write_to_db_empty_lists():
    with pytest.raises(ValueError):
        write_to_db(
            predictions=[],
            directions=[],
            model_name="model_test",
            prediction_date="2023-01-01",
            ticker="AAPL",
        )


def test_write_to_db_mismatched_lengths():
    with pytest.raises(ValueError):
        write_to_db(
            predictions=[1.0],
            directions=[0, 1],
            model_name="model_test",
            prediction_date="2023-01-01",
            ticker="AAPL",
        )
