
"""W3 Z3: Ridge and Lasso regularization with alpha search."""

import mlflow
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso

from .problems import (
    EXPERIMENTS,
    RIDGE_ALPHAS,
    LASSO_ALPHAS,
    LASSO_MAX_ITER,
    load_problem,
    search_alpha,
    plot_search,
    plot_path,
    standardized,
    fit_and_log,
)
from .weights import choose_alpha, weight_summary, top_weights


def setup(experiment_name):
    """Configure MLflow tracking and select an experiment."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)


def run_model(problem_name, model_name):
    """Search alpha, select the best model, and log its results.

    The test set is deliberately not used here.
    """
    if problem_name not in EXPERIMENTS:
        raise ValueError(f"Unknown problem: {problem_name}")

    if model_name not in ("ridge", "lasso"):
        raise ValueError(f"Unknown model: {model_name}")

    data = load_problem(problem_name)
    experiment_name = EXPERIMENTS[problem_name]
    setup(experiment_name)

    if model_name == "ridge":
        make_model = lambda alpha: Ridge(alpha=alpha)
        alphas = RIDGE_ALPHAS
    else:
        make_model = lambda alpha: Lasso(
            alpha=alpha,
            max_iter=LASSO_MAX_ITER,
        )
        alphas = LASSO_ALPHAS

    search, path = search_alpha(make_model, alphas, data)
    chosen_alpha = choose_alpha(search)

    with mlflow.start_run(run_name=f"{model_name}-search"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("chosen_alpha", chosen_alpha)

        mlflow.log_table(search, "alpha_search.json")

        search_fig = plot_search(search, chosen_alpha)
        mlflow.log_figure(search_fig, "alpha_search.png")
        plt.close(search_fig)

        path_fig = plot_path(path, chosen_alpha)
        mlflow.log_figure(path_fig, "weight_path.png")
        plt.close(path_fig)

    final_estimator = make_model(chosen_alpha)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("alpha", chosen_alpha)

        model, weights = fit_and_log(
            standardized(final_estimator),
            data,
        )

        mlflow.log_metrics(weight_summary(weights))
        mlflow.log_dict(top_weights(weights), "top_weights.json")

    print(f"Problem: {problem_name}")
    print(f"Model: {model_name}")
    print(f"Chosen alpha: {chosen_alpha}")
    print(f"Best validation MSE: {search.loc[search['val_mse'].idxmin(), 'val_mse']:.2f}")
    print(f"Nonzero weights: {weight_summary(weights)['nonzero']}")

    return model, weights, chosen_alpha


def run_all():
    """Run Ridge and Lasso for both W3 problems."""
    results = {}

    for problem_name in EXPERIMENTS:
        for model_name in ("ridge", "lasso"):
            results[(problem_name, model_name)] = run_model(
                problem_name,
                model_name,
            )

    return results


if __name__ == "__main__":
    run_all()
