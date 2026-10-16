import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from data import (
    load_frame,
    split_data,
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

LOG_FEATURES = [
    "MedInc",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
]

CLIP_FEATURES = [
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
]


def transform_none(train, val):
    return (
        train[FEATURES].copy(),
        val[FEATURES].copy(),
    )


def transform_log1p(train, val):
    X_train = train[FEATURES].copy()
    X_val = val[FEATURES].copy()

    for col in LOG_FEATURES:
        X_train[col] = np.log1p(X_train[col])
        X_val[col] = np.log1p(X_val[col])

    return X_train, X_val


def transform_clip_q99(train, val):
    X_train = train[FEATURES].copy()
    X_val = val[FEATURES].copy()

    for col in CLIP_FEATURES:
        threshold = X_train[col].quantile(0.99)

        X_train[col] = X_train[col].clip(
            upper=threshold
        )

        X_val[col] = X_val[col].clip(
            upper=threshold
        )

    return X_train, X_val


def transform_poly(train, val, degree):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_val = scaler.transform(
        val[FEATURES]
    )

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_train = poly.fit_transform(
        X_train
    )

    X_val = poly.transform(
        X_val
    )

    return X_train, X_val


def build_variant(train, val, variant):
    if variant == "none":
        return transform_none(train, val)

    if variant == "log1p":
        return transform_log1p(train, val)

    if variant == "clip-q99":
        return transform_clip_q99(train, val)

    if variant == "poly2":
        return transform_poly(
            train,
            val,
            degree=2,
        )

    if variant == "poly3":
        return transform_poly(
            train,
            val,
            degree=3,
        )

    raise ValueError(variant)


def train_variant(train, val, variant):
    X_train, X_val = build_variant(
        train,
        val,
        variant,
    )

    y_train = train["MedHouseVal"]
    y_val = val["MedHouseVal"]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(X_val)

    mse = mean_squared_error(
        y_val,
        pred,
    )

    return (
        model,
        mse,
        X_train,
        X_val,
    )


def main():
    mlflow.set_experiment(
        "w01-transforms"
    )

    df = load_frame()

    train, val, test = split_data(df)

    variants = [
        "none",
        "log1p",
        "clip-q99",
        "poly2",
        "poly3",
    ]

    for variant in variants:

        with mlflow.start_run(
            run_name=variant
        ):
            mlflow.set_tag(
                "stage",
                "selection",
            )

            mlflow.log_param(
                "variant",
                variant,
            )

            model, mse, _, _ = train_variant(
                train,
                val,
                variant,
            )

            mlflow.log_metric(
                "val_mse",
                mse,
            )

            print(
                f"{variant}: {mse:.4f}"
            )

    runs = mlflow.search_runs(
        experiment_names=[
            "w01-transforms"
        ],
        filter_string=(
            "tags.stage = 'selection'"
        ),
        order_by=[
            "metrics.val_mse ASC"
        ],
        max_results=1,
    )

    best_variant = runs.iloc[0][
        "params.variant"
    ]

    print(
        f"Best variant: {best_variant}"
    )

    full_train = train.copy()

    X_train_t, X_test_t = build_variant(
        full_train,
        test,
        best_variant,
    )

    y_train = train["MedHouseVal"]
    y_test = test["MedHouseVal"]

    model = LinearRegression()

    model.fit(
        X_train_t,
        y_train,
    )

    with mlflow.start_run(
        run_name=f"final-test-{best_variant}"
    ):

        mlflow.set_tag(
            "stage",
            "final",
        )

        mlflow.set_tag(
            "test_evaluated",
            "true",
        )

        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train_t[:5]
            if not hasattr(
                X_train_t,
                "head",
            )
            else X_train_t.head(5),
            registered_model_name="california-linear",
        )

        client = MlflowClient()

        client.set_registered_model_alias(
            "california-linear",
            "champion",
            info.registered_model_version,
        )

        test_ds = mlflow.data.from_pandas(
            test,
            name="california-test",
            targets="MedHouseVal",
        )

        mlflow.log_input(
            test_ds,
            context="testing",
        )

        champion = (
            mlflow.pyfunc.load_model(
                "models:/california-linear@champion"
            )
        )

        test_pred = champion.predict(
            X_test_t
        )

        test_mse = mean_squared_error(
            y_test,
            test_pred,
        )

        mlflow.log_metric(
            "test_mse",
            test_mse,
        )

        print(
            f"Test MSE: {test_mse:.4f}"
        )


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
        )
    main()