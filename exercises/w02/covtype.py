
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

from mlproject.evaluation.metrics import multiclass_metrics


EXPERIMENT_NAME = "w02-covtype"
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000


def log_metrics(prefix, metric_dict):
    for name, value in metric_dict.items():
        if name != "recall_per_class":
            mlflow.log_metric(f"{prefix}_{name}", value)


def main():
    # Load Covertype dataset
    dataset = fetch_covtype()

    X = dataset.data
    y = dataset.target

    # Stratified sample of 100,000 rows
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Split 60 / 20 / 20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sample,
        y_sample,
        test_size=0.4,
        stratify=y_sample,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Baseline: majority class
    majority_class = int(np.bincount(y_train).argmax())

    with mlflow.start_run(run_name="baseline-majority"):
        mlflow.log_param("model", "majority-class")
        mlflow.log_param("majority_class", majority_class)

        baseline_predictions = np.full(
            len(y_val), majority_class
        )

        baseline_metrics = multiclass_metrics(
            y_val, baseline_predictions
        )

        log_metrics("val", baseline_metrics)

        print("Baseline validation:", baseline_metrics)

    # Logistic Regression
    with mlflow.start_run(run_name="logreg"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("class_weight", "None")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ])

        model.fit(X_train, y_train)

        # Validation predictions
        val_predictions = model.predict(X_val)

        val_metrics = multiclass_metrics(
            y_val, val_predictions
        )

        log_metrics("val", val_metrics)

        # Recall per class
        recall_dict = {
            str(class_label): float(recall)
            for class_label, recall in zip(
                np.unique(y),
                val_metrics["recall_per_class"],
            )
        }

        mlflow.log_dict(
            recall_dict,
            "recall_per_class.json",
        )

        # Normalized confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_val,
            val_predictions,
            normalize="true",
            values_format=".2f",
        )

        plt.title("Validation Confusion Matrix")
        plt.savefig(
            "validation_confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        mlflow.log_artifact(
            "validation_confusion_matrix.png"
        )

        # Final test evaluation - only once
        test_predictions = model.predict(X_test)

        test_metrics = multiclass_metrics(
            y_test, test_predictions
        )

        log_metrics("test", test_metrics)

        mlflow.set_tag("test_evaluated", "true")

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train[:5],
        )

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
