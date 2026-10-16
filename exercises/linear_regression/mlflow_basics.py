from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.pyfunc
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import load_frame, split_data


def main():
    mlflow.set_experiment("w01-mlflow-basics")

    with mlflow.start_run(run_name="basics"):
        df = load_frame()
        train, val, test = split_data(df)

        # 1. Log artifacts

        mlflow.log_artifact(
            "exercises/linear_regression/data.py",
            artifact_path="code",
        )

        mlflow.log_dict(
            {
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
            },
            "split.json",
        )

        # 2. Log datasets

        dataset_train = mlflow.data.from_pandas(
            train,
            name="california-train",
            targets="MedHouseVal",
        )

        mlflow.log_input(
            dataset_train,
            context="training",
        )

        dataset_train_minus_one = mlflow.data.from_pandas(
            train.iloc[1:],
            name="california-train-minus-first-row",
            targets="MedHouseVal",
        )

        print("Original digest:", dataset_train.digest)
        print("Modified digest:", dataset_train_minus_one.digest)

        # 3. Train LinearRegression on MedInc only

        X_train = train[["MedInc"]]
        y_train = train["MedHouseVal"]

        X_val = val[["MedInc"]]
        y_val = val["MedHouseVal"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save as plain joblib file

        Path("data").mkdir(exist_ok=True)

        model_path = "data/medinc.joblib"

        joblib.dump(model, model_path)

        loaded_model = joblib.load(model_path)

        pred_before = model.predict(X_val)
        pred_after = loaded_model.predict(X_val)

        assert (pred_before == pred_after).all()

        mlflow.log_artifact(
            model_path,
            artifact_path="model-file",
        )

        # Validation MSE

        mse = mean_squared_error(
            y_val,
            pred_before,
        )

        print(f"Validation MSE: {mse:.4f}")

        mlflow.log_metric(
            "val_mse",
            mse,
        )

        # 4. Log and register model in MLflow

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="california-medinc",
        )

        print("MLflow model URI:", model_info.model_uri)

    # 5. Load model from registry

    registry_model = mlflow.pyfunc.load_model(
        "models:/california-medinc/1"
    )

    predictions = registry_model.predict(
        X_val.iloc[:3]
    )

    print("\nPredictions from registry model:")
    print(predictions)


if __name__ == "__main__":
    mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
    )
    main()