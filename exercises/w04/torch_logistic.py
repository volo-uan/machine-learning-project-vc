
"""W4: train PyTorch logistic regression with and without L2 regularization."""

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from previous import breast_cancer_w2, weights_table
from mlproject.training.trainer import train_one_epoch, evaluate


EPOCHS = 1000
LEARNING_RATE = 0.1
BATCH_SIZE = 32
SEED = 42
C = 1.0


def calculate_val_auc(model, dataset):
    """Calculate ROC AUC on the complete validation dataset."""
    X_val, y_val = dataset.tensors

    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probabilities = torch.sigmoid(logits).cpu().numpy().ravel()

    y_true = y_val.cpu().numpy().ravel()

    return roc_auc_score(y_true, probabilities)


def run_experiment(w2, run_name, use_l2):
    """Train one model and log its metrics and weights to MLflow."""

    # Reset the seed so both experiments start with identical weights.
    torch.manual_seed(SEED)

    model = nn.Linear(30, 1)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        w2.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    train_eval_loader = DataLoader(
        w2.train,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    val_loader = DataLoader(
        w2.val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    weight_decay = 0.0

    if use_l2:
        # sklearn objective:
        # mean(log-loss) + 1 / (2 * C * n) * ||weights||^2
        # PyTorch SGD weight_decay adds this L2 term to the gradient.
        weight_decay = 1.0 / (C * len(w2.train))

        optimizer = torch.optim.SGD(
            [
                {"params": model.weight, "weight_decay": weight_decay},
                {"params": model.bias, "weight_decay": 0.0},
            ],
            lr=LEARNING_RATE,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "nn.Linear(30, 1)",
                "optimizer": "SGD",
                "learning_rate": LEARNING_RATE,
                "loss_function": "BCEWithLogitsLoss",
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "seed": SEED,
                "regularization": "L2" if use_l2 else "none",
                "C": C if use_l2 else "none",
                "weight_decay": weight_decay,
            }
        )

        best_val_loss = float("inf")
        best_val_epoch = 0

        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, train_loader, loss_fn, optimizer)

            # Evaluate every epoch to find the true minimum validation loss.
            train_loss = evaluate(model, train_eval_loader, loss_fn)
            val_loss = evaluate(model, val_loader, loss_fn)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

            # Send metrics to MLflow only once every 10 epochs.
            if epoch % 10 == 0:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    step=epoch,
                )

            if epoch % 100 == 0:
                print(
                    f"{run_name} | "
                    f"Epoch {epoch:04d}/{EPOCHS} | "
                    f"Train loss: {train_loss:.6f} | "
                    f"Val loss: {val_loss:.6f}"
                )

        # Calculate final model statistics.
        final_weights = weights_table(w2, model)

        val_auc = calculate_val_auc(model, w2.val)
        max_abs_weight = final_weights["torch"].abs().max()
        max_abs_weight_diff = final_weights["diff"].abs().max()

        # Save the complete weights comparison table.
        mlflow.log_table(
            data=final_weights,
            artifact_file="weights.json",
        )

        # Save final metrics.
        mlflow.log_metrics(
            {
                "val_auc": val_auc,
                "max_abs_weight": max_abs_weight,
                "max_abs_weight_diff": max_abs_weight_diff,
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
                "final_val_loss": val_loss,
            }
        )

        print(f"\nCompleted: {run_name}")
        print(f"Validation AUC:       {val_auc:.6f}")
        print(f"Max absolute weight:  {max_abs_weight:.6f}")
        print(f"Max weight diff:      {max_abs_weight_diff:.6f}")
        print(f"Best validation loss:{best_val_loss:.6f}")
        print(f"Best epoch:           {best_val_epoch}")
        print(f"Final validation loss:{val_loss:.6f}")

        print("\nFinal weights:")
        print(final_weights)


def main():
    w2 = breast_cancer_w2()

    run_experiment(
        w2=w2,
        run_name="no-penalty",
        use_l2=False,
    )

    run_experiment(
        w2=w2,
        run_name="l2-as-in-w2",
        use_l2=True,
    )


if __name__ == "__main__":
    main()
