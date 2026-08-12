import numpy as np
import pandas as pd

from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting
)


def create_test_data():
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "tenure": rng.integers(1, 72, 100),
        "MonthlyCharges": rng.normal(65, 20, 100),
        "TotalCharges": rng.normal(2200, 1500, 100)
    })

    y = pd.Series(
        rng.integers(0, 2, 100)
    )

    return X, y


def test_logistic_regression_trains():
    X, y = create_test_data()

    model = train_logistic_regression(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_logistic_regression_binary_predictions():
    X, y = create_test_data()

    model = train_logistic_regression(X, y)

    predictions = model.predict(X)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_random_forest_trains():
    X, y = create_test_data()

    model = train_random_forest(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_random_forest_binary_predictions():
    X, y = create_test_data()

    model = train_random_forest(X, y)

    predictions = model.predict(X)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_gradient_boosting_trains():
    X, y = create_test_data()

    model = train_gradient_boosting(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_gradient_boosting_binary_predictions():
    X, y = create_test_data()

    model = train_gradient_boosting(X, y)

    predictions = model.predict(X)

    assert set(np.unique(predictions)).issubset({0, 1})
