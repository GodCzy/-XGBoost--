import pandas as pd
from data_preprocessing import preprocess


def test_preprocess_returns_expected_shapes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "emission": [7, 8, 9]})
    X, y, scaler, report = preprocess(df)
    assert X.shape[0] == 3
    assert list(y) == [7, 8, 9]
    assert "missing_rate" in report
    assert scaler is not None
