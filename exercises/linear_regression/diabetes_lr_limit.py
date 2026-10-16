import mlflow
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from linear_regression import (
    standardize,
    add_intercept,
    fit_gradient_descent,
    fit_normal_equation,
    mse,
)

SEED = 42
MAX_ITER = 20_000
TOL = 1e-6


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("w01-diabetes")

    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    Z_train, Z_test = standardize(
        X_train,
        X_test,
    )

    A_train = add_intercept(Z_train)
    A_test = add_intercept(Z_test)

    w_ne = fit_normal_equation(
        A_train,
        y_train,
    )

    target_mse = mse(
        y_train,
        A_train @ w_ne,
    )

    n = len(A_train)

    gram = (A_train.T @ A_train) / n

    eigenvalues = np.linalg.eigvalsh(gram)

    lambda_max = eigenvalues.max()

    lr_limit = 1.0 / lambda_max

    learning_rates = [
        lr_limit * 0.10,
        lr_limit * 0.50,
        lr_limit * 0.90,
        lr_limit * 0.99,
        lr_limit * 1.01,
        lr_limit * 1.20,
    ]

    print(f"lambda_max = {lambda_max:.6f}")
    print(f"lr_limit   = {lr_limit:.6f}")

    with mlflow.start_run(run_name="lr-limit"):

        mlflow.log_param(
            "lr_limit",
            lr_limit,
        )

        mlflow.log_param(
            "lambda_max",
            lambda_max,
        )

        for lr in learning_rates:

            with mlflow.start_run(
                run_name=f"lr-{lr:.6f}",
                nested=True,
            ):

                w, losses, status = fit_gradient_descent(
                    A_train,
                    y_train,
                    lr=lr,
                    max_iter=MAX_ITER,
                    tol=TOL,
                    target_mse=target_mse,
                )

                mlflow.log_param(
                    "lr",
                    lr,
                )

                mlflow.log_metric(
                    "iterations",
                    len(losses),
                )

                mlflow.set_tag(
                    "status",
                    status,
                )

                for step, loss in enumerate(
                    losses,
                    start=1,
                ):
                    mlflow.log_metric(
                        "train_mse",
                        loss,
                        step=step,
                    )

                print(
                    f"lr={lr:.6f} "
                    f"status={status} "
                    f"iterations={len(losses)}"
                )


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
        )
    main()
