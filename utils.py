from typing import Tuple

import numpy as np


def np_weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate weighted acc
    where weight of each sample depends on it's target:
    {rejected_by_product:1, rejected_by_function:1, accepted:2}

    Parameters
    ----------
    y_true: np.ndarray
        True labels

    y_pred: np.ndarray
        Predictions (int)

    Returns
    -------
    weighted_accuracy_score: float
    """
    # 1 for correctly guessed `0` or `1` and 2 for correctly guessed `2`
    weights = np.where(y_true > 1, 2, 1)
    weighted_accuracy_score = np.dot((y_true == y_pred), weights) / weights.sum()
    return weighted_accuracy_score


# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
def lgb_accuracy(y_true, y_pred) -> Tuple[str, float, bool]:
    """
    Calculate weighted acc
    where weight of each sample depends on it's target:
    {rejected_by_product:1, rejected_by_function:1, accepted:2}

    Parameters
    ----------
    y_true: np.ndarray
        True labels

    y_pred: np.ndarray
        Predictions (int)

    Returns
    -------
    result: Tuple
        name: str
        eval_result: float
        is_higher_better: bool
    """
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1)
    y_pred = y_pred.argmax(axis=0)
    y_true = np.int32(y_true)
    weighted_accuracy_score = np_weighted_accuracy(y_true, y_pred)

    return 'weighted_acc', weighted_accuracy_score, True
