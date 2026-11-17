
"""W4, Z2: train a PyTorch linear model and compare it with the W1 reference."""

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from previous import california_w1, weights_table
from mlproject.training.trainer import train_one_epoch, evaluate

def main():
    # 1. Load W1 data and reference model
    w1 = california_w1()

    # 2. Reproducible model initialization
    torch.manual_seed(42)
    model = nn.Linear(8, 1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 3. DataLoaders
    train_loader = DataLoader(
        w1.train,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    train_eval_loader = DataLoader(
        w1.train,
        batch_size=256,
        shuffle=False,
    )

    val_loader = DataLoader(
        w1.val,
        batch_size=256,
        shuffle=False,
    )

    # 4. Start one MLflow run
    with mlflow.start_run(run_name="torch-sgd"):

        # Log training parameters
        mlflow.log_params({
            "model": "nn.Linear(8, 1)",
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "loss_function": "MSELoss",
            "batch_size": 256,
            "epochs": 100,
            "seed": 42,
            "reference_source": w1.source,
        })

        mlflow.set_tag("model_type", "torch-sgd")

        # 5. Train for 100 epochs
        for epoch in range(1, 101):

            train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )

            # Evaluate on complete train and validation datasets
            train_mse = evaluate(
                model,
                train_eval_loader,
                loss_fn,
            )

            val_mse = evaluate(
                model,
                val_loader,
                loss_fn,
            )

            # Compare PyTorch weights with sklearn reference weights
            table = weights_table(w1, model)
            max_abs_weight_diff = table["diff"].abs().max()

            # Log metrics with epoch as step
            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "max_abs_weight_diff": max_abs_weight_diff,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:03d}/100 | "
                f"Train MSE: {train_mse:.6f} | "
                f"Val MSE: {val_mse:.6f} | "
                f"Max weight diff: {max_abs_weight_diff:.6f}"
            )

        # 6. Log final weights table
        final_weights = weights_table(w1, model)

        mlflow.log_table(
            data=final_weights,
            artifact_file="weights.json",
        )

        # 7. Log W1 reference metrics for comparison
        mlflow.log_metrics({
            "reference_train_mse": w1.reference["train_mse"],
            "reference_val_mse": w1.reference["val_mse"],
        })

        print("\nTraining completed!")
        print("\nReference metrics:")
        print(w1.reference)

        print("\nFinal weights:")
        print(final_weights)

if __name__ == "__main__":
    main()