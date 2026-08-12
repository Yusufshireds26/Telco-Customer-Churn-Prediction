import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model


def create_test_data():
    X = pd.DataFrame({
        "feature1": [
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9
        ],

        "feature2": [
            1, 1, 2, 2, 3,
            3, 4, 4, 5, 5
        ]
    })

    y = pd.Series([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    return X, y


def test_evaluate_model_columns():
    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    results = evaluate_model(
        model,
        X,
        y,
        "Test Model"
    )

    expected_columns = {
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    }

    assert expected_columns.issubset(
        set(results.columns)
    )


def test_evaluation_metrics_in_valid_range():
    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    results = evaluate_model(
        model,
        X,
        y,
        "Test Model"
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]

    for metric in metrics:
        value = results.iloc[0][metric]

        assert 0 <= value <= 1


def test_model_name_preserved():
    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    results = evaluate_model(
        model,
        X,
        y,
        "Logistic Regression"
    )

    assert (
        results.iloc[0]["Model"]
        == "Logistic Regression"
    )
