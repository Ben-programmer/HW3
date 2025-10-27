import pandas as pd
from data.preprocess import preprocess_df


def test_preprocess_basic():
    df = pd.DataFrame({"label": ["spam", "ham"], "message": ["WIN $100 now!!!", "hello friend"]})
    out = preprocess_df(df)
    assert "label" in out.columns and "message" in out.columns
    assert out.shape[0] == 2
    assert out.iloc[0]["label"] == "spam"
