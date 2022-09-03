"""
This module defines the transform step of the pipeline

`transform_fn`: defines customizable logic for transforming input data before it is passed to the estimator during model inference.
"""

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler

def calculate_features(df: DataFrame):
    """

    """
    df['rent'] = df['rent']

    return df

def transformer_fn():
    """

    """
    steps = [
        (
            "minmax_scaler",
            MinMaxScaler(),
            ["rent"]
        )
    ]
    return Pipeline(steps)
