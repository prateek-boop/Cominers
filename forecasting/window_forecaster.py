"""Causal forecasting of closed, completed-flow windows.

A prior for window j is corrected only by the observation of window j. Missing
windows reset the prior; they are never filled with invented observations.
"""
import numpy as np
import torch
from torch import nn
from forecasting.delta_predictor import DeltaPredictor
from forecasting.kalman_filter import StateKalmanFilter

STATE_SCHEMA = 'mean_memory_active_completed_flows_v1'


class NormalizedDeltaPredictor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.predictor = DeltaPredictor(state_dim=state_dim, hidden_dim=hidden_dim)
        self.register_buffer('mean', torch.zeros(state_dim))
        self.register_buffer('scale', torch.ones(state_dim))

    def forward(self, states):
        outputs = self.predictor((states-self.mean)/self.scale.clamp_min(1e-6))
        return {key: value*self.scale + (self.mean if key.startswith('forecast') else 0)
                for key, value in outputs.items()}


class WindowForecaster:
    def __init__(self, state_dim, encoder_sha256, checkpoint=None, window_seconds=15):
        self.state_dim = state_dim
        self.encoder_sha256 = encoder_sha256
        self.window_seconds = window_seconds
        if window_seconds != 15:
            raise ValueError('The three forecast heads require 15-second windows')
        self.model = None
        self.previous_window = None
        self.pending = {}
        self.kalman = StateKalmanFilter(state_dim)
        self.validation = None
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if (ckpt.get('kind') != 'latent_window_forecast_v1' or
                ckpt.get('encoder_sha256') != encoder_sha256 or
                ckpt.get('state_schema') != STATE_SCHEMA or
                ckpt.get('flow_batch_size') != 1 or
                ckpt.get('window_seconds') != window_seconds or ckpt.get('state_dim') != state_dim):
                raise ValueError('Forecast checkpoint does not match the encoder/window/state protocol')
            model = NormalizedDeltaPredictor(state_dim, ckpt['hidden_dim'])
            model.load_state_dict(ckpt['state_dict'], strict=True)
            if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
                raise ValueError('Nonfinite forecast checkpoint')
            self.validation = ckpt['validation']
            if not self.validation.get('beats_persistence_all_horizons', False):
                raise ValueError('Forecast candidate failed validation against persistence')
            self.model = model.eval()

    def observe_window(self, window_index, observed_state, observed_at):
        value = np.asarray(observed_state, dtype=np.float32)
        if value.shape != (self.state_dim,) or not np.isfinite(value).all():
            raise ValueError('Expected one finite latent state of the configured dimension')
        if not isinstance(window_index, (int, np.integer)) or not np.isfinite(observed_at):
            raise ValueError('Invalid observation clock')
        if observed_at < (window_index+1)*self.window_seconds:
            raise ValueError('Window must be closed before forecasting')
        if self.previous_window is not None and window_index <= self.previous_window:
            raise ValueError('Closed windows must be strictly chronological')
        contiguous = self.previous_window is not None and window_index == self.previous_window+1
        corrected = False
        prior_error = None
        if not contiguous:
            self.pending.clear()
            self.kalman = StateKalmanFilter(self.state_dim)
        anchor = value
        if contiguous and window_index in self.pending:
            prior = self.pending[window_index]
            prior_error = float(np.mean((prior-value)**2))
            anchor = self.kalman.reanchor_and_correct(prior, value)
            corrected = True
        self.previous_window = int(window_index)
        self.pending.clear()
        result = dict(status='untrained', window_index=int(window_index),
                      state_schema=STATE_SCHEMA, window_seconds=self.window_seconds,
                      observed_at=float(observed_at), reanchored=corrected,
                      prior_mse=prior_error, gap_reset=not contiguous,
                      forecasts=[], deployment_approved=False)
        if self.model is None:
            return result
        with torch.no_grad():
            forecasts = self.model(torch.from_numpy(anchor).unsqueeze(0))
        for step in (1, 2, 3):
            state = forecasts[f'forecast_{step*15}s'][0].numpy()
            if not np.isfinite(state).all():
                raise ValueError('Nonfinite latent forecast')
            self.pending[window_index+step] = state.copy()
            result['forecasts'].append(dict(target_window=window_index+step,
                                            target_window_end=(window_index+step+1)*self.window_seconds,
                                            horizon_seconds=step*self.window_seconds,
                                            latent_state=state.tolist()))
        result.update(status='trained_candidate', validation=self.validation)
        return result
