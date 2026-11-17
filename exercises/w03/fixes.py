
"""W3 Z2: feature-selection fixes for the noise and cholesterol problems."""

import mlflow
import pandas as pd

from sklearn.linear_model import LinearRegression

from .problems import (
    EXPERIMENTS,
    load_problem,
    standardized,
    fit_and_log,
)
from .ols import correlations
from .weights import weight_summary, top_weights


CORRELATION_THRESHOLD = 0.1


def select_features(data, threshold=CORRELATION_THRESHOLD):
    """Select features using absolute training-set correlation with the target.

    Feature names are not used to decide which columns to select.
    """
    feature_correlations = correlations(data.X_train, data.y_train)

    return [
        feature
        for feature, correlation in feature_correlations.items()
        if abs(correlation) > threshold
    ]


def run_fix(problem_name, threshold=CORRELATION_THRESHOLD):
    """Run correlation-based feature selection and log the result to MLflow."""
    if problem_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown problem {problem_name!r}. "
            f"Choose one of: {list(EXPERIMENTS)}"
        )

    data = load_problem(problem_name)

    selected_features = select_features(data, threshold)

    experiment_name = EXPERIMENTS[problem_name]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    run_name = f"fix-correlation-{threshold:g}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("fix_method", "correlation_threshold")
        mlflow.log_param("correlation_threshold", threshold)
        mlflow.log_param("n_selected_features", len(selected_features))

        model, weights = fit_and_log(
            standardized(LinearRegression()),
            data,
            columns=selected_features,
        )

        if weights is not None:
            mlflow.log_metrics(weight_summary(weights))
            mlflow.log_dict(top_weights(weights), "top_weights.json")

        mlflow.log_dict(
            {
                "selected_features": selected_features,
                "correlation_threshold": threshold,
            },
            "selected_features.json",
        )

    print(f"Problem: {problem_name}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Features: {selected_features}")

    return model, weights


def run_noise():
    return run_fix("noise")


def run_cholesterol():
    return run_fix("cholesterol")


if __name__ == "__main__":
    run_noise()
    run_cholesterol()
