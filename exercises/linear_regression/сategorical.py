import numpy as np
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import (
    load_frame,
    split_data,
    add_region,
    one_hot,
)


NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def fit_and_score(train, val, features):
    y_train = train["MedHouseVal"]
    y_val = val["MedHouseVal"]

    X_train = train[features].copy()
    X_val = val[features].copy()

    numeric_cols = [
        c
        for c in features
        if not c.startswith("region_")
    ]

    if numeric_cols:
        scaler = StandardScaler()

        X_train[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols]
        )

        X_val[numeric_cols] = scaler.transform(
            X_val[numeric_cols]
        )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(X_val)

    return mean_squared_error(
        y_val,
        pred,
    )


def run_model(
    run_name,
    train,
    val,
    features,
    encoding,
):
    with mlflow.start_run(
        run_name=run_name,
        nested=True,
    ):
        mse = fit_and_score(
            train,
            val,
            features,
        )

        mlflow.log_param(
            "features",
            ",".join(features),
        )

        mlflow.log_param(
            "n_features",
            len(features),
        )

        mlflow.log_param(
            "encoding",
            encoding,
        )

        mlflow.log_metric(
            "val_mse",
            mse,
        )

        print(
            f"{run_name}: "
            f"{mse:.4f}"
        )


def main():
    mlflow.set_experiment(
        "w01-categorical"
    )

    df = add_region(
        load_frame()
    )

    train, val, test = split_data(df)

    with mlflow.start_run(
        run_name="region-onehot"
    ):
        counts = (
            train["region"]
            .value_counts()
            .to_dict()
        )

        mlflow.log_dict(
            counts,
            "region_counts.json",
        )

        # full one-hot

        train_full, _, full_cols = one_hot(
            train,
            train,
            "region",
            drop_first=False,
        )

        X_full = train_full[
            full_cols
        ].to_numpy()

        X_full = np.column_stack(
            [
                np.ones(
                    len(X_full)
                ),
                X_full,
            ]
        )

        rank_full = np.linalg.matrix_rank(
            X_full
        )

        mlflow.log_metric(
            "rank_full",
            rank_full,
        )

        # drop-first one-hot

        train_drop, _, drop_cols = one_hot(
            train,
            train,
            "region",
            drop_first=True,
        )

        X_drop = train_drop[
            drop_cols
        ].to_numpy()

        X_drop = np.column_stack(
            [
                np.ones(
                    len(X_drop)
                ),
                X_drop,
            ]
        )

        rank_drop = np.linalg.matrix_rank(
            X_drop
        )

        mlflow.log_metric(
            "rank_drop",
            rank_drop,
        )

        print(
            "rank_full =",
            rank_full,
        )

        print(
            "rank_drop =",
            rank_drop,
        )

        # datasets for modelling

        train_reg, val_reg, region_cols = one_hot(
            train,
            val,
            "region",
            drop_first=True,
        )

        train_reg_full, val_reg_full, region_cols_full = one_hot(
            train,
            val,
            "region",
            drop_first=False,
        )

        no_location = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
        ]

        run_model(
            "all",
            train,
            val,
            NUMERIC_FEATURES,
            "none",
        )

        run_model(
            "region-only",
            train_reg,
            val_reg,
            region_cols,
            "drop_first",
        )

        run_model(
            "no-location+region",
            train_reg,
            val_reg,
            no_location + region_cols,
            "drop_first",
        )

        run_model(
            "all+region",
            train_reg,
            val_reg,
            NUMERIC_FEATURES + region_cols,
            "drop_first",
        )

        run_model(
            "all+region-full",
            train_reg_full,
            val_reg_full,
            NUMERIC_FEATURES + region_cols_full,
            "full",
        )


if __name__ == "__main__":
    mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
    )
    main()