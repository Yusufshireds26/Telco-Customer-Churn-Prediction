from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data_processing import (
    load_data,
    clean_data,
    prepare_features
)

from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting
)

from src.evaluation import evaluate_model


def main():

    data_file = (
        PROJECT_ROOT
        / "data"
        / "telco_churn.csv"
    )

    data = load_data(data_file)

    data = clean_data(data)

    X_train, X_test, y_train, y_test = prepare_features(data)

    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    gradient_boosting_model = train_gradient_boosting(
        X_train,
        y_train
    )

    results = pd.concat([
        evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression"
        ),

        evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "Random Forest"
        ),

        evaluate_model(
            gradient_boosting_model,
            X_test,
            y_test,
            "Gradient Boosting"
        )
    ])

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = (
        results_dir
        / "model_metrics.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print()
    print("Model Training Complete")
    print()
    print(results)
    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
