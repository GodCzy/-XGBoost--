from pathlib import Path

import pandas as pd
import pytest

from data_preprocessing import (
    build_feature_frame,
    clean_data,
    frame_from_records,
    load_dataset,
    preprocess,
)


def test_preprocess_returns_expected_shapes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "emission": [7, 8, 9]})
    X, y, scaler, report = preprocess(df)
    assert X.shape[0] == 3
    assert list(y) == [7, 8, 9]
    assert "missing_rate" in report
    assert scaler is not None
    assert "feature_summary" in report
    assert "high_correlation_pairs" in report
    assert report["retained_features"]


def test_clean_data_handles_single_row():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    cleaned = clean_data(df)
    assert not cleaned.empty


def test_build_feature_frame_generates_engineered_features():
    df = pd.DataFrame({"a": [1.0, 2.0], "emission": [3.0, 4.0]})
    frame = build_feature_frame(df)
    assert "a_sq" in frame.columns


def test_load_dataset_detects_csv(tmp_path: Path):
    sample = pd.DataFrame(
        {
            "electricity": [100, 120],
            "GDP总量(亿元)": [50, 60],
            "emission": [80, 95],
        }
    )
    dataset = tmp_path / "sample.csv"
    sample.to_csv(dataset, index=False)
    loaded = load_dataset(dataset)
    assert "gdp_total" in loaded.columns
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        sample.rename(columns={"GDP总量(亿元)": "gdp_total"}),
    )


def test_frame_from_records_validates_payload():
    records = [
        {"electricity": 100, "GDP总量(亿元)": 55, "emission": 70},
        {"electricity": 110, "gdp_total": 60, "emission": 82},
    ]
    frame = frame_from_records(records)
    assert "gdp_total" in frame.columns
    assert len(frame) == 2


def test_frame_from_records_requires_emission():
    with pytest.raises(ValueError):
        frame_from_records([{"electricity": 1}])
