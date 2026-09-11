"""Phase 6: True Calibration via Split Conformal Prediction.

Provides finite-sample coverage guarantees (e.g. 90% coverage for alpha=0.10):
    Non-conformity score: s_i = 1 - P(y_i | x_i)
    Quantile threshold: q_hat = Quantile(s_1, ..., s_n; ceil((n+1)(1 - alpha)) / n)
    Conformal Prediction Set: C(X_test) = { y in Y : 1 - P(y | X_test) <= q_hat }
If the model encounters ambiguous traffic, the set automatically expands (e.g. [Lateral Movement, C2]).
"""
from typing import Dict, List, Optional, Set, Any
import numpy as np


class MultiClassSplitConformal:
    """Split Conformal Predictor for multi-class MITRE ATT&CK stages and binary alerts."""

    def __init__(self, alpha: float = 0.10, stage_names: Optional[List[str]] = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.stage_names = stage_names or [
            "benign", "reconnaissance", "initial_access", "credential_access",
            "lateral_movement", "command_and_control", "exfiltration", "impact"
        ]
        self.calibration_scores: np.ndarray = np.array([])
        self.q_hat: float = 0.85
        self.is_calibrated: bool = False

    def calibrate(self, y_true_stages: np.ndarray, y_prob_matrix: np.ndarray):
        """
        Calibrate non-conformity scores on hold-out calibration set:
            s_i = 1 - P(y_i | x_i)
        """
        y_true = np.asarray(y_true_stages, dtype=np.int64).flatten()
        probs = np.asarray(y_prob_matrix, dtype=np.float64)

        if probs.ndim != 2 or not len(y_true) or len(y_true) != len(probs):
            raise ValueError("Size mismatch between y_true and probability matrix")

        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(axis=1), 1):
            raise ValueError("Expected normalized finite probability rows")
        if np.any((y_true < 0) | (y_true >= probs.shape[1])):
            raise ValueError("Calibration label outside class range")
        # Extract predicted probability assigned to the ground-truth class
        n = len(y_true)
        true_class_probs = probs[np.arange(n), y_true]
        scores = 1.0 - true_class_probs
        self.calibration_scores = np.sort(scores)

        # Finite-sample adjusted quantile level: ceil((n+1)(1 - alpha)) / n
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        self.q_hat = float(self.calibration_scores[rank - 1]) if rank <= n else 1.0
        self.num_classes = probs.shape[1]
        self.is_calibrated = True

    def predict_conformal_set(self, prob_vector: np.ndarray) -> Dict[str, Any]:
        """
        Constructs prediction set C(X_test) = { y in Y : 1 - P(y | X) <= q_hat }
        """
        probs = np.asarray(prob_vector, dtype=np.float64).flatten()
        if not probs.size or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or not np.isclose(probs.sum(), 1):
            raise ValueError("Expected a normalized finite probability vector")
        if self.is_calibrated and probs.size != self.num_classes:
            raise ValueError("Probability vector does not match calibrated class count")
        q_threshold = self.q_hat if self.is_calibrated else 1.0

        # Inclusion condition: P(y | X) >= 1 - q_hat
        min_p_needed = max(0.0, 1.0 - q_threshold)

        included_indices = [idx for idx, p in enumerate(probs) if p >= min_p_needed]
        # Guarantee non-empty set: fallback to argmax if set would be empty
        if not included_indices:
            included_indices = [int(np.argmax(probs))]

        included_stages = [
            self.stage_names[idx] if idx < len(self.stage_names) else f"stage_{idx}"
            for idx in included_indices
        ]

        # Check if critical stages are in the conformal set (Lateral Movement or C2)
        has_critical_stage = any(
            s in ("lateral_movement", "command_and_control", "impact") for s in included_stages
        )

        return {
            "q_hat": float(q_threshold),
            "alpha": float(self.alpha),
            "is_calibrated": self.is_calibrated,
            "coverage_guarantee": f"{int((1 - self.alpha) * 100)}% under exchangeability" if self.is_calibrated else None,
            "conformal_set": included_stages,
            "set_size": len(included_stages),
            "is_singleton": len(included_stages) == 1,
            "has_critical_stage": has_critical_stage,
        }
