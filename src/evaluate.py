"""Metric helpers shared across baseline and DeBERTa evaluation."""
import numpy as np
from sklearn.metrics import log_loss

from src.data import TARGET_COLS


def multiclass_log_loss(y_true_df, y_pred_proba: np.ndarray) -> float:
    """y_true_df: DataFrame with the one-hot TARGET_COLS.
    y_pred_proba: array of shape (n, 3) in the same column order as TARGET_COLS.
    """
    y_true = y_true_df[TARGET_COLS].to_numpy().argmax(axis=1)
    return log_loss(y_true, y_pred_proba, labels=[0, 1, 2])


def assert_valid_probabilities(proba: np.ndarray, atol: float = 1e-3):
    assert not np.isnan(proba).any(), "NaNs in predicted probabilities"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilities out of [0, 1] range"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=atol), f"rows do not sum to 1: min={row_sums.min()}, max={row_sums.max()}"
