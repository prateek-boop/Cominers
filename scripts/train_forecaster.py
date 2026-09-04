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

import os
import glob
from torch.utils.data import TensorDataset, DataLoader

def load_checkpoint(checkpoint_path, model, optimizer):
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming forecaster from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    return start_epoch

if __name__ == "__main__":
    print("Starting Forecaster Training Process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    STATE_DIM = 128
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    CHECKPOINT_PATH = "data/processed/forecaster_checkpoint.pt"
    
    # Initialize Model
    model = DeltaPredictor(STATE_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    
    # Resume from checkpoint
    start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    
    # Load Dataset (Expects tensors of node embeddings sequences and targets)
    # E.g. sequences.pt shape (N, seq_len, STATE_DIM), targets.pt shape (N, STATE_DIM)
    seq_file = "data/processed/forecaster_sequences.pt"
    if not os.path.exists(seq_file):
        print(f"No processed data found at {seq_file}. Please generate sequences first.")
        print("Exiting...")
        exit(0)
    
    print("Loading data...")
    dataset_dict = torch.load(seq_file)
    sequences = dataset_dict['sequences']
    t15s = dataset_dict['t15s']
    t30s = dataset_dict['t30s']
    t45s = dataset_dict['t45s']
    
    dataset = TensorDataset(sequences, t15s, t30s, t45s)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        
        for batch_seq, b_t15, b_t30, b_t45 in loader:
            batch_seq = batch_seq.to(device)
            optimizer.zero_grad()
            
            preds = model(batch_seq)
            
            loss_15 = criterion(preds['15s'], b_t15.to(device))
            loss_30 = criterion(preds['30s'], b_t30.to(device))
            loss_45 = criterion(preds['45s'], b_t45.to(device))
            
            loss = loss_15 + loss_30 + loss_45
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")
