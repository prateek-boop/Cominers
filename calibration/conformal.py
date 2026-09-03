import numpy as np
from typing import List

class SplitConformalPredictor:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat = 0.0
        
    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        n = len(cal_labels)
        scores = 1.0 - cal_probs[np.arange(n), cal_labels]
        
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        if q_level > 1.0:
            self.q_hat = np.max(scores)
        else:
            self.q_hat = np.quantile(scores, q_level, method='higher')
            
    def predict_set(self, test_probs: np.ndarray, classes: List[str]) -> List[List[str]]:
        prediction_sets = []
        for probs in test_probs:
            mask = probs >= (1.0 - self.q_hat)
            pred_set = [classes[i] for i, m in enumerate(mask) if m]
            prediction_sets.append(pred_set)
        return prediction_sets
