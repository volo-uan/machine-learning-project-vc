
from sklearn import metrics


def binary_metrics(y_true, p, threshold):
    """Calculate binary classification metrics.

    p >= threshold means positive class (1).
    """
    y_pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(metrics.recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(metrics.f1_score(
            y_true, y_pred, zero_division=0
        )),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def choose_threshold(y_true, p, thresholds, min_recall):
    """Choose the largest threshold satisfying minimum recall."""
    best_threshold = None

    for threshold in thresholds:
        result = binary_metrics(y_true, p, threshold)

        if result["recall"] >= min_recall:
            if best_threshold is None or threshold > best_threshold:
                best_threshold = threshold

    return best_threshold


def multiclass_metrics(y_true, y_pred):
    """Calculate multiclass classification metrics."""
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "f1_macro": float(metrics.f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "f1_micro": float(metrics.f1_score(
            y_true, y_pred, average="micro", zero_division=0
        )),
        "f1_weighted": float(metrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "recall_per_class": metrics.recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
    }
