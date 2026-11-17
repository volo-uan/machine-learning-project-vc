
import time

import mlflow
import numpy as np
import torch

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mlproject.models.mlp import MLP
from mlproject.evaluation.metrics import multiclass_metrics


EXPERIMENT_NAME = "w05-covtype-mlp"
BASELINE_EXPERIMENT_NAME = "w02-covtype"

SEED = 42
SAMPLE_SIZE = 100_000
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
N_FEATURES = 54
N_CLASSES = 7

ARCHITECTURES = {
    "linear": (),
    "256": (256,),
    "256-256": (256, 256),
}


def prepare_data():
    """Load, sample, split and standardize the Covertype dataset."""

    dataset = fetch_covtype()

    X = dataset.data
    y = dataset.target

    # Stratified sample of 100,000 rows, identical to W2.
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=SEED,
    )

    # Split 60 / 20 / 20, identical to W2.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sample,
        y_sample,
        test_size=0.4,
        stratify=y_sample,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=SEED,
    )

    # Standardization fitted only on training data.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert labels from 1-7 to 0-6 for CrossEntropyLoss.
    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    print(f"Training rows: {len(train_dataset)}")
    print(f"Validation rows: {len(val_dataset)}")
    print(f"Test rows: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, shuffle=False):
    """Create a DataLoader with a reproducible generator."""

    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(model, loader, loss_fn, optimizer):
    """Train the model for one epoch and return average loss."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, loss_fn):
    """Calculate average loss and collect predictions."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    average_loss = total_loss / total_samples

    return (
        average_loss,
        np.array(all_targets),
        np.array(all_predictions),
    )


def get_logreg_accuracy():
    """Read validation accuracy from the W2 Logistic Regression run."""

    experiment = mlflow.get_experiment_by_name(
        BASELINE_EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{BASELINE_EXPERIMENT_NAME}' not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'logreg'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            "Logistic Regression run 'logreg' not found in W2."
        )

    accuracy_column = "metrics.val_accuracy"

    if accuracy_column not in runs.columns:
        raise RuntimeError(
            "The W2 Logistic Regression run has no val_accuracy metric."
        )

    accuracy = float(runs.iloc[0][accuracy_column])

    print(f"W2 Logistic Regression validation accuracy: {accuracy:.4f}")

    return accuracy


def log_validation_metrics(y_true, y_pred):
    """Log W2 multiclass metrics and per-class recall."""

    metrics = multiclass_metrics(y_true, y_pred)

    mlflow.log_metrics({
        "val_f1_macro": metrics["f1_macro"],
        "val_f1_micro": metrics["f1_micro"],
        "val_f1_weighted": metrics["f1_weighted"],
    })

    # Original Covertype labels are 1-7.
    recall_dict = {
        str(class_label): float(recall)
        for class_label, recall in zip(
            range(1, N_CLASSES + 1),
            metrics["recall_per_class"],
        )
    }

    mlflow.log_dict(
        recall_dict,
        "recall_per_class.json",
    )

    print("Validation metrics:", metrics)

    return metrics


def train_architecture(
    name,
    hidden,
    train_dataset,
    val_dataset,
    search_id,
    logreg_accuracy,
):
    """Train one MLP architecture and log its validation results."""

    torch.manual_seed(SEED)

    model = MLP(
        N_FEATURES,
        N_CLASSES,
        hidden=hidden,
    )

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    with mlflow.start_run(run_name=f"search-{name}"):

        mlflow.log_params({
            "architecture_name": name,
            "hidden": str(hidden),
            "in_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "n_parameters": n_parameters,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss_function": "CrossEntropyLoss",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
        })

        mlflow.set_tags({
            "stage": "search",
            "search_id": search_id,
        })

        # Save the W2 Logistic Regression accuracy for comparison.
        mlflow.log_metric(
            "logreg_val_accuracy",
            logreg_accuracy,
        )

        for epoch in range(1, EPOCHS + 1):

            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )

            val_loss, y_true, y_pred = evaluate(
                model,
                val_loader,
                loss_fn,
            )

            val_accuracy = float(
                np.mean(y_true == y_pred)
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch)

            print(
                f"{name} | Epoch {epoch:02d}/{EPOCHS} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_accuracy={val_accuracy:.4f}"
            )

        # Final validation metrics and sensitivity per class.
        _, y_true, y_pred = evaluate(
            model,
            val_loader,
            loss_fn,
        )

        validation_metrics = log_validation_metrics(
            y_true,
            y_pred,
        )

        final_accuracy = validation_metrics["accuracy"]

        mlflow.log_metric(
            "final_val_accuracy",
            final_accuracy,
        )

        print(
            f"{name}: parameters={n_parameters}, "
            f"final validation accuracy={final_accuracy:.4f}"
        )

    return model


def select_best_architecture(search_id, experiment_id):
    """Select the best architecture using validation accuracy."""

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.search_id = '{search_id}' "
            "AND tags.stage = 'search'"
        ),
        order_by=["metrics.val_accuracy DESC"],
        max_results=3,
    )

    if runs.empty:
        raise RuntimeError("No search runs found in MLflow.")

    print("\nArchitecture comparison:")

    print(
        runs[
            [
                "tags.mlflow.runName",
                "params.hidden",
                "params.n_parameters",
                "metrics.val_accuracy",
                "metrics.logreg_val_accuracy",
            ]
        ].to_string(index=False)
    )

    best_run = runs.iloc[0]

    best_run_id = best_run["run_id"]
    best_name = best_run["params.architecture_name"]

    print(f"\nSelected architecture: {best_name}")
    print(
        f"Best validation accuracy: "
        f"{best_run['metrics.val_accuracy']:.4f}"
    )

    return best_run_id, best_name


def evaluate_final_model(
    model,
    best_run_id,
    best_name,
    test_dataset,
    search_id,
    logreg_accuracy,
):
    """Evaluate the selected model on the test set exactly once."""

    test_loader = make_loader(test_dataset, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"final-test-{best_name}"):

        mlflow.set_tags({
            "stage": "final",
            "test_evaluated": "true",
            "search_id": search_id,
        })

        mlflow.log_params({
            "selected_architecture": best_name,
            "selected_run_id": best_run_id,
        })

        mlflow.log_metric(
            "logreg_val_accuracy",
            logreg_accuracy,
        )

        test_loss, y_true, y_pred = evaluate(
            model,
            test_loader,
            loss_fn,
        )

        test_metrics = multiclass_metrics(
            y_true,
            y_pred,
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_micro": test_metrics["f1_micro"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        })

        print(f"\nFinal test loss ({best_name}): {test_loss:.4f}")
        print("Final test metrics:", test_metrics)


def main():

    mlflow.set_tracking_uri("http://localhost:5000")

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    search_id = time.strftime("%Y%m%d-%H%M%S")

    print(f"Search ID: {search_id}")

    logreg_accuracy = get_logreg_accuracy()

    train_dataset, val_dataset, test_dataset = prepare_data()

    # Stage 1: train and compare all three architectures.
    trained_models = {}

    for name, hidden in ARCHITECTURES.items():

        trained_models[name] = train_architecture(
            name,
            hidden,
            train_dataset,
            val_dataset,
            search_id,
            logreg_accuracy,
        )

    # Stage 2: select the best architecture using validation accuracy.
    best_run_id, best_name = select_best_architecture(
        search_id,
        experiment.experiment_id,
    )

    # Stage 3: evaluate only the selected model on the test set.
    best_model = trained_models[best_name]

    evaluate_final_model(
        best_model,
        best_run_id,
        best_name,
        test_dataset,
        search_id,
        logreg_accuracy,
    )


if __name__ == "__main__":
    main()
