from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

REGIONS = {
    "LA": (34.05, -118.24),
    "SF": (37.77, -122.42),
    "SD": (32.72, -117.16),
    "SJ": (37.34, -121.89),
    "SAC": (38.58, -121.49),
}

import pandas as pd


def one_hot(train, other, column, drop_first=True):
    train = train.copy()
    other = other.copy()

    categories = sorted(train[column].unique())

    if drop_first:
        categories = categories[1:]

    new_columns = []

    for category in categories:
        col_name = f"{column}_{category}"

        train[col_name] = (
            train[column] == category
        ).astype(int)

        other[col_name] = (
            other[column] == category
        ).astype(int)

        new_columns.append(col_name)

    train = train.drop(columns=[column])
    other = other.drop(columns=[column])

    return train, other, new_columns

def add_region(df):
    df = df.copy()

    regions = []

    for _, row in df.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]

        distances = {
            name: np.sqrt(
                (lat - center_lat) ** 2
                + (lon - center_lon) ** 2
            )
            for name, (center_lat, center_lon) in REGIONS.items()
        }

        nearest = min(distances, key=distances.get)

        if distances[nearest] < 0.75:
            regions.append(nearest)
        else:
            regions.append("other")

    df["region"] = regions

    return df


def load_frame():
    """
    Return the California Housing dataset as a single DataFrame
    with 8 features and the target column 'MedHouseVal'.
    """
    dataset = fetch_california_housing(as_frame=True)
    return dataset.frame


def split_data(df, seed=42):
    """
    Split the full DataFrame into train/validation/test
    in proportions 60/20/20.

    Returns:
        train_df, val_df, test_df
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,  # 25% of 80% = 20% of total
        random_state=seed,
    )

    return train_df, val_df, test_df

