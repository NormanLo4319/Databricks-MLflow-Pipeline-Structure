import pytest
import os
import tempfile
import pandas as pd
from pandas import DataFrame
from steps.ingest import loadFileAsDataframe

@pytest.fixture
def sample_data():
    return pd.read_parquet(
        os.path.join(os.path.dirname(__file__), ".parquet")
    )

def test_ingest_function_reads_csv_correctly(sample_data):
    tempdir = tempfile.mkdtemp()
    csv_path = os.path.join(tempdir, ".csv")
    sample_data.to_csv(csv_path)
    ingested = loadFileAsDataframe(csv_path, "csv")
    assert isinstance(ingested, DataFrame)