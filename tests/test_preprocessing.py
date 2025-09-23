import pandas as pd

from data_preprocessing import build_feature_frame, clean_data, preprocess


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
