import numpy as np
from numpy.typing import ArrayLike


def _validate_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1D float arrays and verify matching shapes."""
    true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    if true_arr.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {true_arr.shape}, y_pred has {pred_arr.shape}."
        )
    if np.isnan(true_arr).any() or np.isnan(pred_arr).any():
        raise ValueError("NaN values detected in y_true or y_pred.")

    return true_arr, pred_arr


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error (symmetric)."""
    true_arr, pred_arr = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))


def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error (average error in flight cycles)."""
    true_arr, pred_arr = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(pred_arr - true_arr)))


def compute_nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float]:
    """Compute asymmetric NASA C-MAPSS RUL score (total and per-sample mean).

    Penalizes late predictions (y_pred > y_true) more heavily than early ones.
    Lower is better; 0.0 is perfect.
    """
    true_arr, pred_arr = _validate_inputs(y_true, y_pred)
    diff = pred_arr - true_arr

    # Vectorized piecewise exponential penalty
    penalties = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0,
    )
    total_score = float(np.sum(penalties))
    mean_score = float(np.mean(penalties))
    return total_score, mean_score


def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute all evaluation metrics in a flat dict ready for MLflow logging."""
    true_arr, pred_arr = _validate_inputs(y_true, y_pred)
    nasa_total, nasa_mean = compute_nasa_score(true_arr, pred_arr)
    overpred_rate = float(np.mean(pred_arr > true_arr))

    return {
        "rmse": compute_rmse(true_arr, pred_arr),
        "mae": compute_mae(true_arr, pred_arr),
        "nasa_score_total": nasa_total,
        "nasa_score_mean": nasa_mean,
        "overprediction_rate": overpred_rate,
    }