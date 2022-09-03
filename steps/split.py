"""
This module defines the `split` steps of the pipeline

- `processSplits`: defines customizable logic for processing and cleaning the training, validation,
and test datasets produced by the data splitting procedure.
"""

from pandas import DataFrame

def processSplits(train_df: DataFrame, validation_df:DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """


    """

    def process(df: DataFrame):
        # Fill missing data and drop invalid data points
        filledDF = df
        droppedDF = filledDF.dropna()

        # Filter out outliers
        cleanedDF = droppedDF.iloc[()]

        return cleanedDF

    return process(train_df), process(validation_df), process(test_df)