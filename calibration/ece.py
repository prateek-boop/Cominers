"""Phase 6: Expected Calibration Error (ECE) & Reliability Assessment.

Measures the alignment between model confidence probabilities and true empirical accuracy:
ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
"""
from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
    n_bins: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Expected Calibration Error and per-bin confidence vs accuracy stats.
    """
    if n_bins is not None:
        num_bins = n_bins

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError("num_bins must be positive")
    if y_true.ndim != 1 or y_true.shape != y_prob.shape or not y_true.size:
        raise ValueError("Expected nonempty matching label and probability vectors")
    if not np.isin(y_true, [0, 1]).all() or not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.minimum(np.digitize(y_prob, bins) - 1, num_bins - 1)

    ece = 0.0
    bin_stats = []

    for i in range(num_bins):
        mask = (bin_indices == i)
        count = int(np.sum(mask))

        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            ece += (count / len(y_true)) * abs_diff

            bin_stats.append({
                "bin_index": i,
                "bin_range": (float(bins[i]), float(bins[i + 1])),
                "sample_count": count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(abs_diff, 4),
            })
        else:
            bin_stats.append({
                "bin_index": i,
                "bin_range": (float(bins[i]), float(bins[i + 1])),
                "sample_count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "num_bins": num_bins,
        "n_bins": num_bins,
        "bins": bin_stats,
        "bin_stats": bin_stats,
        "is_well_calibrated": bool(ece < 0.05),
    }


# Convenience alias
compute_ece = compute_expected_calibration_error
