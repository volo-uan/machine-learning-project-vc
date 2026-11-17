
import torch


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One pass over loader with a step per batch."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for X, y in loader:
        # 1. Clear gradients from previous batch
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(X)

        # 3. Calculate loss
        loss = loss_fn(predictions, y)

        # 4. Backpropagation
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

        # 6. Accumulate loss weighted by batch size
        batch_size = X.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model, loader, loss_fn):
    """Mean loss over all examples without gradients."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for X, y in loader:
            predictions = model(X)
            loss = loss_fn(predictions, y)

            batch_size = X.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples
