
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

from mlproject.evaluation.metrics import binary_metrics, choose_threshold


EXPERIMENT_NAME = "w02-breast-cancer"
RANDOM_STATE = 42
MIN_RECALL = 0.98
THRESHOLDS = np.arange(0.05, 1.00, 0.05)


def main():
    # Load data
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = (dataset.target == 0).astype(int)

    # Split 60 / 20 / 20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Data description
    description = {
        "dataset": "Breast Cancer Wisconsin",
        "target_positive_class": "malignant",
        "target_encoding": "target == 0 means malignant (1)",
        "train": {
            "samples": len(X_train),
            "malignant": int(y_train.sum()),
        },
        "validation": {
            "samples": len(X_val),
            "malignant": int(y_val.sum()),
        },
        "test": {
            "samples": len(X_test),
            "malignant": int(y_test.sum()),
        },
        "features": {
            "count": X.shape[1],
            "types": {
                str(k): int(v)
                for k, v in X.dtypes.astype(str).value_counts().items()
            },
        },
        "missing_values": int(X.isna().sum().sum()),
    }
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic-regression"):
        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("min_recall", MIN_RECALL)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log dataset description
        mlflow.log_dict(description, "data_description.json")

        # Register training and validation datasets
        train_dataset = mlflow.data.from_pandas(
            X_train.assign(target=y_train),
            name="breast-cancer-train",
        )

        val_dataset = mlflow.data.from_pandas(
            X_val.assign(target=y_val),
            name="breast-cancer-validation",
        )

        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        # Build model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

        model.fit(X_train, y_train)

        # Validation probabilities
        val_probabilities = model.predict_proba(X_val)[:, 1]

        # Choose threshold on validation set
        threshold = choose_threshold(
            y_val,
            val_probabilities,
            THRESHOLDS,
            MIN_RECALL,
        )

        if threshold is None:
            raise RuntimeError("No threshold satisfies the required recall.")

        mlflow.log_param("threshold", float(threshold))

        # Validation metrics
        val_metrics = binary_metrics(
            y_val, val_probabilities, threshold
        )

        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)

        val_auc = roc_auc_score(y_val, val_probabilities)
        mlflow.log_metric("val_auc", val_auc)

        # ROC curve artifact
        RocCurveDisplay.from_predictions(
            y_val, val_probabilities
        )
        plt.title("Validation ROC Curve")
        plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact("roc_curve.png")

        # Final test evaluation - only once
        test_probabilities = model.predict_proba(X_test)[:, 1]

        test_metrics = binary_metrics(
            y_test, test_probabilities, threshold
        )

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        mlflow.set_tag("test_evaluated", "true")

        # Test confusion matrix
        test_predictions = (
            test_probabilities >= threshold
        ).astype(int)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            test_predictions,
            display_labels=["Benign", "Malignant"],
        )

        plt.title("Test Confusion Matrix")
        plt.savefig(
            "test_confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        mlflow.log_artifact("test_confusion_matrix.png")

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(5),
        )

        print("Validation metrics:", val_metrics)
        print("Validation AUC:", val_auc)
        print("Selected threshold:", threshold)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
