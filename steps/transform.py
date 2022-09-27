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
    df['RentPerSqFt'] = df['Rent'] / df['MaxSqFt']
    df = df[['RentPerSqFt', 'NumBeds', 'NumBaths', 'MaxSqFt', 'BuildingRatingID']]

    return df

def transformer_fn():
    """

    """
    return Pipeline(
        steps = [
            (
                "calculate_features",
                FunctionTransformer(calculate_features, feature_names_out='one-to-one'),
            ),
            (
                "encoder",
                ColumnTransformer(
                    Transformers=[
                    (
                        "std_scaler",
                        StandardScaler(),
                        ["Rent"],
                    ),
                    ]
                ),
            ),
        ]
    )
