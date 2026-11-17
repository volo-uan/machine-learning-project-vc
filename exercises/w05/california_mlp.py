
"""W5 Z2: Compare a linear model with MLP architectures on California Housing."""

import time

import mlflow
import mlflow.pytorch
import pandas as pd
import torch

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mlproject.models.mlp import MLP, count_parameters
from mlproject.training.trainer import train_one_epoch, evaluate


SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.003

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

TARGET = "MedHouseVal"

ARCHITECTURES = {
    "linear": (),
    "64": (64,),
    "64-64": (64, 64),
    "256-256": (256, 256),
}


def prepare_data():
    """Load, split and standardize California Housing data."""

    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame

    # First split: 80% remaining, 20% test.
    rest, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=SEED,
    )

    # Second split: 75% train, 25% validation from the remaining data.
    train, val = train_test_split(
        rest,
        test_size=0.25,
        random_state=SEED,
    )

    # Fit the scaler only on training data.
    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    def make_tensor_dataset(data):
        X = scaler.transform(data[FEATURES])
        y = data[TARGET].values.reshape(-1, 1)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return TensorDataset(X, y)

    train_dataset = make_tensor_dataset(train)
    val_dataset = make_tensor_dataset(val)
    test_dataset = make_tensor_dataset(test)

    print(f"Training rows: {len(train_dataset)}")
    print(f"Validation rows: {len(val_dataset)}")
    print(f"Test rows: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, shuffle):
    """Create a DataLoader with a reproducible generator."""

    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def train_architecture(name, hidden, train_dataset, val_dataset, search_id):
    """Train one architecture and log its results to MLflow."""

    torch.manual_seed(SEED)

    model = MLP(
        in_features=8,
        n_classes=1,
        hidden=hidden,
    )

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    n_parameters = count_parameters(model)

    with mlflow.start_run(run_name=f"search-{name}"):

        mlflow.log_params({
            "architecture_name": name,
            "hidden": str(hidden),
            "in_features": 8,
            "n_classes": 1,
            "n_parameters": n_parameters,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
        })

        mlflow.set_tags({
            "stage": "search",
            "search_id": search_id,
        })

        for epoch in range(1, EPOCHS + 1):

            train_mse = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )

            val_mse = evaluate(
                model,
                val_loader,
                loss_fn,
            )

            mlflow.log_metrics({
                "train_mse": train_mse,
                "val_mse": val_mse,
            }, step=epoch)

        # Save the trained model so the selected architecture
        # can be loaded later without training it again.
        

        print(
            f"{name}: parameters={n_parameters}, "
            f"final validation MSE={val_mse:.4f}"
        )
        return model


def select_best_architecture(search_id, experiment_id):
    """Select the best run using validation MSE from MLflow."""

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.search_id = '{search_id}' "
            "AND tags.stage = 'search'"
        ),
        order_by=["metrics.val_mse ASC"],
        max_results=4,
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
                "metrics.val_mse",
            ]
        ].to_string(index=False)
    )

    best_run = runs.iloc[0]

    best_run_id = best_run["run_id"]
    best_name = best_run["params.architecture_name"]

    print(f"\nSelected architecture: {best_name}")
    print(f"Best validation MSE: {best_run['metrics.val_mse']:.4f}")

    return best_run_id, best_name


def evaluate_final_model(
    model,
    best_run_id,
    best_name,
    test_dataset,
    search_id,
):
    """Evaluate the selected model on the test set exactly once."""

    test_loader = make_loader(test_dataset, shuffle=False)
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name=f"final-test-{best_name}"):

        mlflow.set_tags({
            "stage": "final",
            "test_evaluated": "true",
            "search_id": search_id,
        })

        mlflow.log_param("selected_architecture", best_name)
        mlflow.log_param("selected_run_id", best_run_id)

        test_mse = evaluate(
            model,
            test_loader,
            loss_fn,
        )

        mlflow.log_metric("test_mse", test_mse)

        print(f"\nFinal test MSE ({best_name}): {test_mse:.4f}")

def main():

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment("w05-california-mlp")

    # One unique identifier for all architecture searches in this execution.
    search_id = time.strftime("%Y%m%d-%H%M%S")

    print(f"Search ID: {search_id}")

    train_dataset, val_dataset, test_dataset = prepare_data()

    # Stage 1: train and compare all four architectures.
        # Stage 1: train and compare all four architectures.
    trained_models = {}

    for name, hidden in ARCHITECTURES.items():

        trained_models[name] = train_architecture(
            name,
            hidden,
            train_dataset,
            val_dataset,
            search_id,
        )

    # Stage 2: select the best architecture using MLflow.
    best_run_id, best_name = select_best_architecture(
        search_id,
        experiment.experiment_id,
    )

    # Stage 3: evaluate only the selected model on the test set.
        # Stage 3: evaluate only the selected model on the test set.
    best_model = trained_models[best_name]

    evaluate_final_model(
        best_model,
        best_run_id,
        best_name,
        test_dataset,
        search_id,
    )


if __name__ == "__main__":
    main()
