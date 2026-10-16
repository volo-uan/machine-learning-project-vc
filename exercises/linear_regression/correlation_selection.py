import json

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import load_frame, split_data


EXPERIMENT_NAME = "w01-correlation"
TARGET = "MedHouseVal"


def train_and_evaluate(train, val, features):
    X_train = train[features]
    y_train = train[TARGET]

    X_val = val[features]
    y_val = val[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return mean_squared_error(y_val, predictions)


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_frame()

    train, val, test = split_data(df)

    train_ds = mlflow.data.from_pandas(
        train,
        name="california-train",
        targets=TARGET,
    )

    val_ds = mlflow.data.from_pandas(
        val,
        name="california-validation",
        targets=TARGET,
    )

    corr_matrix = train.corr(numeric_only=True)

    target_corr = (
        corr_matrix[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda x: abs(x), ascending=False)
    )

    with mlflow.start_run(run_name="correlation-selection"):

        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(val_ds, context="validation")

        mlflow.log_dict(
            {k: float(v) for k, v in target_corr.items()},
            "correlation_with_target.json",
        )

        # correlation figure

        fig, ax = plt.subplots(figsize=(10, 8))

        image = ax.imshow(
            corr_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(
            corr_matrix.columns,
            rotation=90,
        )

        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_yticklabels(corr_matrix.columns)

        plt.colorbar(image)

        plt.tight_layout()

        mlflow.log_figure(
            fig,
            "correlation_train.png",
        )

        plt.close(fig)

        thresholds = [0, 0.05, 0.10, 0.15, 0.30]

        for threshold in thresholds:

            selected_features = [
                feature
                for feature, corr in target_corr.items()
                if abs(corr) >= threshold
            ]

            mse = train_and_evaluate(
                train,
                val,
                selected_features,
            )

            with mlflow.start_run(
                run_name=f"threshold-{threshold:.2f}",
                nested=True,
            ):
                mlflow.log_param(
                    "threshold",
                    threshold,
                )

                mlflow.log_param(
                    "n_features",
                    len(selected_features),
                )

                mlflow.log_param(
                    "features",
                    ",".join(selected_features),
                )

                mlflow.log_metric(
                    "val_mse",
                    mse,
                )

                print(
                    f"threshold={threshold:.2f} "
                    f"features={len(selected_features)} "
                    f"mse={mse:.4f}"
                )


if __name__ == "__main__":
    mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
    )
    main()