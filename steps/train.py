"""
This module defines the `train` step of the pipeline

`estimator_fn`: defines the customizable estimator type and parameters that are used during training to produce a model pipeline
"""

def estimator_fn():
    """

    """
    from sklearn.linear_model import LinearRegression

    return LinearRegression()