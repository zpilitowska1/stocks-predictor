import pandas as pd
import pytest
from stocks_predictor.data_loader import load_and_prepare_data


@pytest.fixture
def sample_csv(tmp_path):
    data = """date,ticker,target
2023-01-01,AAPL,150
2023-01-02,QCOM,200
"""
    file = tmp_path / "sample.csv"
    file.write_text(data)
    return str(file)


def test_load_and_prepare_data_basic(sample_csv):
    df = load_and_prepare_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 2


def test_load_and_prepare_data_empty(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_and_prepare_data(str(empty_file))


def test_load_and_prepare_data_invalid_date(tmp_path):
    data = "date,ticker,target\ninvalid_date,AAPL,150"
    file = tmp_path / "invalid_date.csv"
    file.write_text(data)
    with pytest.raises(ValueError):  # <-- expecting ValueError here
        load_and_prepare_data(str(file))
