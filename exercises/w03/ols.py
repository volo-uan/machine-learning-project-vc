import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from .problems import (
    load_problem,
    standardized,
    fit_and_log,
)

from .weights import (
    weight_summary,
    top_weights,
)


def setup(name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(name)


def correlations(X, y):
    return {
        col: float(X[col].corr(y))
        for col in X.columns
    }


def run_noise():
    data = load_problem("noise")

    setup("w03-noise")

    with mlflow.start_run(run_name="ols"):
        model, weights = fit_and_log(
            standardized(LinearRegression()),
            data
        )

        mlflow.log_metrics(
            weight_summary(weights)
        )

        mlflow.log_dict(
            top_weights(weights),
            "top_weights.json"
        )

        mlflow.log_dict(
            correlations(
                data.X_train,
                data.y_train
            ),
            "correlation_with_target.json"
        )


def run_cholesterol():
    data = load_problem("cholesterol")

    setup("w03-cholesterol")

    with mlflow.start_run(run_name="ols"):
        model, weights = fit_and_log(
            standardized(LinearRegression()),
            data
        )

        mlflow.log_metrics(
            weight_summary(weights)
        )

        mlflow.log_dict(
            top_weights(weights),
            "top_weights.json"
        )

        mlflow.log_dict(
            correlations(
                data.X_train,
                data.y_train
            ),
            "correlation_with_target.json"
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            data.X_train.corr(),
            cmap="coolwarm",
            center=0
        )

        plt.tight_layout()
        plt.savefig("correlation.png")

        mlflow.log_artifact(
            "correlation.png"
        )

        plt.close()


if __name__ == "__main__":
    run_noise()
    run_cholesterol()
