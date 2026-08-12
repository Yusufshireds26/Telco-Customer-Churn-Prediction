import pandas as pd

from src.data_processing import (
    clean_data,
    prepare_features
)


def create_sample_data():
    return pd.DataFrame({
        "customerID": [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J"
        ],

        "tenure": [
            1, 5, 10, 15, 20,
            25, 30, 40, 50, 60
        ],

        "MonthlyCharges": [
            20.0, 30.0, 40.0, 50.0, 60.0,
            70.0, 80.0, 90.0, 100.0, 110.0
        ],

        "TotalCharges": [
            "20",
            "150",
            "400",
            "750",
            "1200",
            "1750",
            "2400",
            "3600",
            "5000",
            "6600"
        ],

        "Contract": [
            "Month-to-month",
            "Month-to-month",
            "One year",
            "One year",
            "Two year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month"
        ],

        "InternetService": [
            "DSL",
            "Fiber optic",
            "DSL",
            "No",
            "Fiber optic",
            "DSL",
            "Fiber optic",
            "No",
            "DSL",
            "Fiber optic"
        ],

        "Churn": [
            "Yes",
            "No",
            "Yes",
            "No",
            "No",
            "Yes",
            "No",
            "No",
            "Yes",
            "No"
        ]
    })


def test_clean_data_converts_churn():
    data = create_sample_data()

    cleaned = clean_data(data)

    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_clean_data_converts_total_charges():
    data = create_sample_data()

    cleaned = clean_data(data)

    assert pd.api.types.is_numeric_dtype(
        cleaned["TotalCharges"]
    )


def test_prepare_features_removes_customer_id():
    data = clean_data(
        create_sample_data()
    )

    X_train, X_test, _, _ = prepare_features(data)

    assert "customerID" not in X_train.columns
    assert "customerID" not in X_test.columns


def test_prepare_features_removes_target():
    data = clean_data(
        create_sample_data()
    )

    X_train, X_test, _, _ = prepare_features(data)

    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns


def test_train_test_split_contains_rows():
    data = clean_data(
        create_sample_data()
    )

    X_train, X_test, y_train, y_test = prepare_features(data)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
