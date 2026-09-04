import pytest
import torch
from forecasting.delta_predictor import DeltaPredictor

def test_delta_predictor_initialization():
    state_dim = 10
    model = DeltaPredictor(state_dim=state_dim, hidden_dim=32, num_layers=1)
    
    assert model.gru.input_size == state_dim
    assert model.gru.hidden_size == 32
    assert model.gru.num_layers == 1
    
    assert model.head_15s.out_features == state_dim
    assert model.head_30s.out_features == state_dim
    assert model.head_45s.out_features == state_dim

def test_delta_predictor_forward():
    state_dim = 10
    seq_len = 5
    batch_size = 4
    
    model = DeltaPredictor(state_dim=state_dim)
    
    # Input shape: (batch_size, sequence_length, state_dim)
    x = torch.randn(batch_size, seq_len, state_dim)
    
    output = model(x)
    
    assert isinstance(output, dict)
    assert '15s' in output
    assert '30s' in output
    assert '45s' in output
    
    # Expected output shape: (batch_size, state_dim)
    expected_shape = (batch_size, state_dim)
    assert output['15s'].shape == expected_shape
    assert output['30s'].shape == expected_shape
    assert output['45s'].shape == expected_shape
