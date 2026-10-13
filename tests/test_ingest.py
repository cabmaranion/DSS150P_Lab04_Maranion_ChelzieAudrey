import pandas as pd

from src.ingest import row_keys


def test_identical_rows_get_distinct_keys():
    df = pd.DataFrame({"a": ["1", "1", "1"], "b": ["0", "0", "0"]})
    keys = row_keys(df)
    assert keys.nunique() == 3
    assert keys.str.endswith("-1").sum() == 1