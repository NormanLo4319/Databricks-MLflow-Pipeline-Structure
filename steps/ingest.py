"""
This module defines the data ingestion step of the pipeline:
Key Function: loadFileAsDataframe: Defines customizable logic for parsing dataset formats that are not
natively parsed by MLFlow Pipelines (i.e. format other than Parquet, Delta, and Spark SQL).
"""

import logging
from pandas import DataFrame

_logger = logging.getLogger(__name__)

def loadFileAsDataframe(filePath: str, fileFormat: str) -> DataFrame:
    """
    Load content from the specific dataset file as a Pandas Dataframe

    filePath: The path to the dataset file or the delta table name
    fileFormat: "csv", "sql", or ...
    return: A Pandas Dataframe representing the content of the specific file.
    """
    if fileFormat == "csv":
        import pandas
        _logger.warning(
            "Loading dataset CSV using `pandas.read_csv()` with default arguments and assumed index"
            " column 0 which may not produce the desired schema. If the schema is not correct, it can be"
            " adjusted by modifying the `loadFileAsDataframe()` function in `steps/ingest.py`"
        )
        return pandas.read_csv(filePath, index_col=0)

    elif fileFormat == "sql":
        _logger.warning(
            "Loading delta table directly into Spark Dataframe, then transform into Pandas Dataframe"
        )
        df = spark.sql(filePath)
        return df.toPandas()

    else:
        raise NotImplementedError