"""W3: what the weights of a fitted model say. Complete weight_summary and top_weights in Z1, choose_alpha in Z3."""

import pandas as pd

from .problems import NOISE_PREFIX


def weight_summary(weights: pd.Series) -> dict:
    """Counts of weights, for mlflow.log_metrics.

    weights: one weight per feature, indexed by the feature name (as returned by fit_and_log).
    Returns a dict with the keys:
        nonzero        - weights different from 0
        zero           - weights exactly equal to 0
        noise_nonzero  - nonzero weights of the noise columns (names starting with NOISE_PREFIX)
        real_nonzero   - nonzero weights of the other columns
    All values are ints.
    """
    nonzero_mask = weights != 0
    noise_mask = weights.index.to_series().str.startswith(NOISE_PREFIX)

    return {
        "nonzero": int(nonzero_mask.sum()),
        "zero": int((~nonzero_mask).sum()),
        "noise_nonzero": int((nonzero_mask & noise_mask.to_numpy()).sum()),
        "real_nonzero": int((nonzero_mask & ~noise_mask.to_numpy()).sum()),
    }


def top_weights(weights: pd.Series, k: int = 5) -> dict:
    """The k weights with the largest absolute value, for mlflow.log_dict.

    Returns {feature name: weight as a float}, the largest |weight| first.
    """
    top = weights.abs().sort_values(ascending=False).head(k)

    return {
        str(name): float(weights[name])
        for name in top.index
    }


def choose_alpha(search: pd.DataFrame) -> float:
    """The alpha with the lowest validation MSE.

    search: a DataFrame with the columns alpha and val_mse (as returned by search_alpha).
    Returns the alpha as a float.
    """
    best_row = search.loc[search["val_mse"].idxmin()]
    return float(best_row["alpha"])