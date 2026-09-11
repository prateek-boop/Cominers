"""Split Conformal Prediction & Conformal Bound for CyberTGN.

Provides distribution-free, finite-sample statistical guarantees for attack alerts.
Given a significance level alpha (e.g. alpha=0.01 for 99% confidence),
bounds the false-alarm rate and produces statistically verified prediction sets.
"""
from typing import List, Optional, Set
import numpy as np


class SplitConformalPredictor:
    """Inductive / Split Conformal Predictor for binary attack classification."""

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (default 0.05 -> 95% coverage guarantee against false alarms).
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.benign_scores: np.ndarray = np.array([])
        self.q_hat: float = 0.85  # default conservative threshold until calibrated
        self.is_calibrated: bool = False

    def calibrate(self, benign_attack_probs: np.ndarray):
        """
        Calibrate using predicted attack probabilities on known benign traffic.
        Non-conformity score S_i = P(attack | X_i_benign).
        """
        if len(benign_attack_probs) == 0:
            raise ValueError("Calibration set cannot be empty")

        values = np.asarray(benign_attack_probs, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Calibration probabilities must be a finite vector in [0, 1]")
        self.benign_scores = np.sort(values)
        n = len(self.benign_scores)

        # Finite-sample adjusted quantile level
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        self.q_hat = float(self.benign_scores[rank - 1]) if rank <= n else 1.0
        self.is_calibrated = True

    def calculate_p_value(self, attack_prob: float) -> float:
        """
        Calculate empirical conformal p-value for the null hypothesis H0: Flow is Benign.
        p_val = (1 + sum(S_i >= attack_prob)) / (n + 1)
        """
        if not self.is_calibrated or len(self.benign_scores) == 0:
            return 1.0

        n = len(self.benign_scores)
        count_higher = np.sum(self.benign_scores >= attack_prob)
        p_val = float((1 + count_higher) / (n + 1))
        return p_val

    def evaluate_flow(self, attack_prob: float, base_threshold: float = 0.85) -> dict:
        """
        Evaluate if flow satisfies the dual criteria:
        1. Model score > base_threshold (e.g. 0.85)
        2. Statistically confirmed by the conformal bound (p-value < alpha)
        """
        p_val = self.calculate_p_value(attack_prob)
        # Conformal bound confirmed when p-value is below significance level alpha
        conformal_confirmed = self.is_calibrated and p_val <= self.alpha

        is_high_risk = bool(attack_prob >= base_threshold and (conformal_confirmed or not self.is_calibrated))

        prediction_set: List[str] = []
        if attack_prob < self.q_hat:
            prediction_set.append("benign")
        if attack_prob >= (1.0 - self.q_hat):
            prediction_set.append("attack")
        if not prediction_set:
            prediction_set = ["attack" if attack_prob >= 0.5 else "benign"]

        return {
            "is_calibrated": self.is_calibrated,
            "attack_prob": float(attack_prob),
            "conformal_q_hat": float(self.q_hat),
            "conformal_p_value": float(p_val),
            "conformal_confirmed": bool(conformal_confirmed),
            "prediction_set": prediction_set,
            "is_high_risk_alert": is_high_risk,
        }
