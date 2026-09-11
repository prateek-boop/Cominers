"""Phase 4: Discrete Kalman Residual Re-Anchoring.

Re-anchors forecasted latent graph states whenever the real network observation
S_{t+1}^{obs} arrives, computing Kalman gain K_t to eliminate error drift:
K_t = Sigma_{t+1|t} (Sigma_{t+1|t} + R)^{-1}
S_{t+1}^{corrected} = S_hat_{t+1|t} + K_t (S_{t+1}^{obs} - S_hat_{t+1|t})
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class StateKalmanFilter:
    """Re-anchors predicted graph states using observed telemetry to prevent drift."""

    def __init__(self, state_dim: int = 128, process_noise: float = 0.05, measurement_noise: float = 2.0):
        self.state_dim = state_dim
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim)

        self.kf.x = np.zeros(state_dim, dtype=np.float32)
        self.kf.F = np.eye(state_dim, dtype=np.float32)
        self.kf.H = np.eye(state_dim, dtype=np.float32)
        self.kf.P *= 100.0
        self.kf.R = np.eye(state_dim, dtype=np.float32) * measurement_noise
        self.kf.Q = np.eye(state_dim, dtype=np.float32) * process_noise

    def reanchor_and_correct(
        self,
        predicted_state: np.ndarray,
        observed_state: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Kalman correction using observed network state:
        Corrects prediction and resets filter anchor point.
        """
        # Ensure flat 1D vectors
        pred_flat = np.asarray(predicted_state, dtype=np.float32).flatten()
        obs_flat = np.asarray(observed_state, dtype=np.float32).flatten()

        if pred_flat.shape != (self.state_dim,) or obs_flat.shape != (self.state_dim,):
            raise ValueError("State vectors must match state_dim")
        if not np.isfinite(pred_flat).all() or not np.isfinite(obs_flat).all():
            raise ValueError("State vectors must be finite")
        # The external forecast supplies the prior mean.
        self.kf.x = pred_flat.copy()
        self.kf.predict()
        self.kf.update(obs_flat)

        # Corrected state S_{t+1}^{corrected}
        corrected = np.array(self.kf.x, dtype=np.float32)
        return corrected

    def predict_next_anchored(self, predicted_delta: np.ndarray) -> np.ndarray:
        """Roll out from current anchored Kalman state."""
        delta_flat = np.asarray(predicted_delta, dtype=np.float32).flatten()
        if delta_flat.shape != (self.state_dim,) or not np.isfinite(delta_flat).all():
            raise ValueError("Delta must be finite and match state_dim")
        self.kf.predict()
        self.kf.x = self.kf.x + delta_flat
        return self.kf.x.copy()
