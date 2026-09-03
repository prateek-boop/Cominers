import torch
from torch.optim import Adam
from torch.nn import MSELoss
from forecasting.delta_predictor import DeltaPredictor

def train_forecaster(sequences, target_deltas_15s, target_deltas_30s, target_deltas_45s, device):
    state_dim = sequences.size(-1)
    model = DeltaPredictor(state_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    
    model.train()
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        preds = model(sequences.to(device))
        
        loss_15 = criterion(preds['15s'], target_deltas_15s.to(device))
        loss_30 = criterion(preds['30s'], target_deltas_30s.to(device))
        loss_45 = criterion(preds['45s'], target_deltas_45s.to(device))
        
        loss = loss_15 + loss_30 + loss_45
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss {loss.item()}")
        
    return model

if __name__ == "__main__":
    print("Forecaster training script ready.")
