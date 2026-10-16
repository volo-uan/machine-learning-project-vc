import mlflow
import mlflow.data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import load_frame, split_data


EXPERIMENT_NAME = "w01-models"


def run_baseline(train, val):
    with mlflow.start_run(run_name="baseline-mean"):
        train_ds = mlflow.data.from_pandas(
            train,
            name="california-train",
            targets="MedHouseVal",
        )
        val_ds = mlflow.data.from_pandas(
            val,
            name="california-val",
            targets="MedHouseVal",
        )

        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(val_ds, context="validation")

        y_train = train["MedHouseVal"]
        y_val = val["MedHouseVal"]

        prediction = y_train.mean()

        y_pred = [prediction] * len(y_val)

        mse = mean_squared_error(y_val, y_pred)

        mlflow.log_param("model", "mean")
        mlflow.log_param("n_features", 0)
        mlflow.log_metric("val_mse", mse)

        print(f"baseline-mean: {mse:.4f}")


def run_linear(train, val, run_name, features):
    with mlflow.start_run(run_name=run_name):
        train_ds = mlflow.data.from_pandas(
            train,
            name="california-train",
            targets="MedHouseVal",
        )
        val_ds = mlflow.data.from_pandas(
            val,
            name="california-val",
            targets="MedHouseVal",
        )

        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(val_ds, context="validation")

        X_train = train[features]
        y_train = train["MedHouseVal"]

        X_val = val[features]
        y_val = val["MedHouseVal"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)

        mlflow.log_param(
            "features",
            ",".join(features),
        )
        mlflow.log_param(
            "n_features",
            len(features),
        )
        mlflow.log_metric(
            "val_mse",
            mse,
        )

        print(f"{run_name}: {mse:.4f}")


def main():
    mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_frame()

    train, val, test = split_data(df)

    all_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    run_baseline(train, val)

    run_linear(
        train,
        val,
        "medinc",
        ["MedInc"],
    )

    run_linear(
        train,
        val,
        "no-location",
        [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
        ],
    )

    run_linear(
        train,
        val,
        "medinc-location",
        [
            "MedInc",
            "Latitude",
            "Longitude",
        ],
    )

    run_linear(
        train,
        val,
        "all",
        all_features,
    )


if __name__ == "__main__":
    main()