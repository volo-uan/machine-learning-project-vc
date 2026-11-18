"""Images from a fixed split (W6: first EuroSAT, then your project data).

The list of (path, label) comes from the fixed split written by your data rules, not from listing
the directory at every run. Check the class with:
    uv run pytest exercises/w06
"""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def read_split(csv_path):
    """Reads a fixed split saved as CSV with the columns path,label. Returns a list of (Path, int)."""
    with open(csv_path, newline="") as f:
        return [(Path(row["path"]), int(row["label"])) for row in csv.DictReader(f)]


class ImageDataset(Dataset):
    """One example on request: opens the image only in __getitem__, never all images at once."""

    def __init__(self, items, transform=None):
        self.items = list(items)  # list of (path, label) from the fixed split
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
