import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    return pd.read_csv(filepath)


def clean_data(data):
    data = data.copy()

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

        data["TotalCharges"] = data["TotalCharges"].fillna(
            data["TotalCharges"].median()
        )

    if "Churn" in data.columns:
        data["Churn"] = data["Churn"].map({
            "Yes": 1,
            "No": 0
        })

    return data


def prepare_features(data):
    data = data.copy()

    X = data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    y = data["Churn"]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    X_train[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_test[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    return X_train, X_test, y_train, y_test
