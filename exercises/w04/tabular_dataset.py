
"""W4 Z4: compare a custom TabularDataset with TensorDataset."""

import copy

import mlflow
import pandas as pd
import torch
import sys
from pathlib import Path
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tabular import Standardize, TabularDataset
from mlproject.training.trainer import train_one_epoch, evaluate

SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.01

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


def prepare_data():
    """Load and split California Housing, fitting scaling on training data."""
    frame = pd.read_csv("data/california_housing.csv")

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(frame), generator=generator).tolist()

    n_test = int(len(frame) * 0.2)
    remaining = indices[n_test:]
    n_val = int(len(remaining) * 0.25)

    test_indices = indices[:n_test]
    val_indices = remaining[:n_val]
    train_indices = remaining[n_val:]

    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    val_frame = frame.iloc[val_indices].reset_index(drop=True)

    transform = Standardize(train_frame, FEATURES)

    train_dataset = TabularDataset(
        train_frame, FEATURES, TARGET, transform
    )
    val_dataset = TabularDataset(
        val_frame, FEATURES, TARGET, transform
    )

    return train_dataset, val_dataset


def make_tensor_datasets(train_dataset, val_dataset):
    """Build TensorDataset objects from the same tensors."""
    train_tensor = TensorDataset(train_dataset.x, train_dataset.y)
    val_tensor = TensorDataset(val_dataset.x, val_dataset.y)

    return train_tensor, val_tensor


def make_loader(dataset, shuffle):
    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def run_experiment(run_name, train_dataset, val_dataset):
    """Train one linear model and log metrics after every epoch."""
    torch.manual_seed(SEED)

    model = nn.Linear(len(FEATURES), 1)
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("w04-california-dataset")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
            "model": "Linear(8, 1)",
        })

        for epoch in range(1, EPOCHS + 1):
            train_mse = train_one_epoch(
                model, train_loader, loss_fn, optimizer
            )
            val_mse = evaluate(model, val_loader, loss_fn)

            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                },
                step=epoch,
            )

        return copy.deepcopy(model.state_dict())


def main():
    train_tabular, val_tabular = prepare_data()

    train_tensor, val_tensor = make_tensor_datasets(
        train_tabular, val_tabular
    )

    print(f"Training rows: {len(train_tabular)}")
    print(f"Validation rows: {len(val_tabular)}")

    tabular_weights = run_experiment(
        "tabular-dataset",
        train_tabular,
        val_tabular,
    )

    tensor_weights = run_experiment(
        "tensor-dataset",
        train_tensor,
        val_tensor,
    )

    max_difference = max(
        (tabular_weights[key] - tensor_weights[key]).abs().max().item()
        for key in tabular_weights
    )

    print(f"Maximum weight difference: {max_difference}")


if __name__ == "__main__":
    main()
