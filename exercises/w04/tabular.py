
"""A Dataset for a table (W4, task Z4): one row on request.

The same three methods as every Dataset, including the image Dataset in W5; only __getitem__ changes.
Check it with:
    uv run pytest exercises/w04
"""

import torch
from torch.utils.data import Dataset


class Standardize:
    """x -> (x - mean) / std, with mean and std computed on the training part only (as StandardScaler in W1)."""

    def __init__(self, frame, features):
        self.mean = torch.tensor(
            frame[features].mean().values,
            dtype=torch.float32
        )
        self.std = torch.tensor(
            frame[features].std(ddof=0).values,
            dtype=torch.float32
        )

    def __call__(self, x):
        return (x - self.mean) / self.std


class TabularDataset(Dataset):
    """Rows of a pandas DataFrame as (features, target) pairs.

    frame: the DataFrame of one part (training, validation or test);
    features: the list of feature columns; target: the name of the target column;
    transform: applied to the feature tensor of every row (e.g. Standardize), or None.
    __getitem__ returns x of shape (len(features),) and y of shape (1,), both float32.
    """

    def __init__(self, frame, features, target, transform=None):
        self.x = torch.tensor(
            frame[features].values,
            dtype=torch.float32
        )
        self.y = torch.tensor(
            frame[[target]].values,
            dtype=torch.float32
        )
        self.transform = transform

        if self.transform is not None:
            self.x = self.transform(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

