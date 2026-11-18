import argparse
import json

import mlflow
import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from mlproject.data.dataset import (
    ImageDataset,
    read_split,
)
from mlproject.models.mlp import (
    MLP,
    count_parameters,
)
from mlproject.models.cnn import SimpleCNN
from mlproject.training.trainer import (
    train_one_epoch,
    evaluate,
)
from mlproject.tracking import setup


BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
SEED = 42

MEAN = [0.3443, 0.3799, 0.4074]
STD = [0.2022, 0.1364, 0.1150]

def shifted_accuracy(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = torch.roll(
                x,
                (16, 16),
                dims=(2, 3),
            )

            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total


def rotated_accuracy(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = torch.rot90(
                x,
                1,
                dims=(2, 3),
            )

            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total

def make_loader(dataset, shuffle=False):
    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=4,
    )


def accuracy(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        required=True,
    )

    args = parser.parse_args()

    setup("w06-eurosat")

    with open("data/eurosat/splits/classes.json") as f:
        class_names = json.load(f)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    train_dataset = ImageDataset(
        read_split("data/eurosat/splits/train.csv"),
        transform=transform,
    )

    val_dataset = ImageDataset(
        read_split("data/eurosat/splits/val.csv"),
        transform=transform,
    )

    test_dataset = ImageDataset(
        read_split("data/eurosat/splits/test.csv"),
        transform=transform,
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        shuffle=False,
    )

    test_loader = make_loader(
        test_dataset,
        shuffle=False,
    )

    torch.manual_seed(SEED)

    if args.model == "mlp":
        model = MLP(
            3 * 64 * 64,
            10,
            hidden=(256,),
        )
    else:
        model = SimpleCNN(10)

    print("parameters:", count_parameters(model))

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LR,
    )

    with mlflow.start_run(run_name=args.model):

        mlflow.log_param(
            "model",
            args.model,
        )

        mlflow.log_param(
            "parameters",
            count_parameters(model),
        )

        for epoch in range(1, EPOCHS + 1):

            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )

            val_loss = evaluate(
                model,
                val_loader,
                loss_fn,
            )

            val_acc = accuracy(
                model,
                val_loader,
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

            print(
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )

            test_acc = accuracy(
                model,
                test_loader,
            )

        shift_acc = shifted_accuracy(
            model,
            val_loader,
        )

        rot_acc = rotated_accuracy(
            model,
            val_loader,
        )

        mlflow.log_metric(
            "val_accuracy_shift16",
            shift_acc,
        )

        mlflow.log_metric(
            "val_accuracy_rot90",
            rot_acc,
        )

        print("shift16:", shift_acc)
        print("rot90  :", rot_acc)    

        mlflow.log_metric(
            "test_accuracy",
            test_acc,
        )

        print("test accuracy:", test_acc)


if __name__ == "__main__":
    main()