import numpy as np
from filterpy.kalman import KalmanFilter

class StateKalmanFilter:
    def __init__(self, state_dim: int):
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim)
        
        self.kf.x = np.zeros(state_dim)
        self.kf.F = np.eye(state_dim)
        self.kf.H = np.eye(state_dim)
        self.kf.P *= 1000.
        self.kf.R = np.eye(state_dim) * 5.0
        self.kf.Q = np.eye(state_dim) * 0.1
        
    def update_and_predict(self, observed_state: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
        if observed_state is not None:
            self.kf.update(observed_state)
            
        self.kf.predict()
        
        corrected_future = self.kf.x + predicted_delta
        return corrected_future
