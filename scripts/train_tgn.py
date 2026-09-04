import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from models.memory import MemoryModule
from models.tgn import CyberTGN
from models.multitask_heads import AttackProbabilityHead, MitreStageClassifier
from torch_geometric.nn.models.tgn import LastNeighborLoader

def train_epoch(loader, memory, gnn, prob_head, mitre_head, neighbor_loader, optimizer, device):
    memory.train()
    gnn.train()
    prob_head.train()
    mitre_head.train()

    memory.reset_state()
    neighbor_loader.reset_state()
    
    bce_loss = BCEWithLogitsLoss()
    ce_loss = CrossEntropyLoss()
    
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        
        neg_dst = torch.randint(0, memory.memory.num_nodes, (src.size(0),), dtype=torch.long, device=device)
        
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        z_mem, last_update = memory(n_id)
        z = gnn(z_mem, last_update, edge_index, batch.t[e_id], batch.msg[e_id])
        
        pos_out = prob_head(z[src], z[pos_dst])
        neg_out = prob_head(z[src], z[neg_dst])
        
        pos_label = (batch.y > 0).float()
        loss_prob_pos = bce_loss(pos_out, pos_label)
        loss_prob_neg = bce_loss(neg_out, torch.zeros_like(neg_out))
        
        mitre_out = mitre_head(z[src], z[pos_dst])
        loss_mitre = ce_loss(mitre_out, batch.y)
        
        loss = loss_prob_pos + loss_prob_neg + loss_mitre
        loss.backward()
        optimizer.step()
        
        memory.detach()
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        total_loss += float(loss) * batch.num_events
        
    return total_loss / len(loader.dataset)

import os
import glob
from torch_geometric.data import TemporalData
from torch.utils.data import DataLoader

def load_checkpoints(checkpoint_path, memory, gnn, prob_head, mitre_head, optimizer):
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        memory.load_state_dict(checkpoint['memory'])
        gnn.load_state_dict(checkpoint['gnn'])
        prob_head.load_state_dict(checkpoint['prob_head'])
        mitre_head.load_state_dict(checkpoint['mitre_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    return start_epoch

if __name__ == "__main__":
    print("Starting TGN Training Process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    NUM_NODES = 100000  # Adjust based on your dataset IP range
    MSG_DIM = 16
    MEM_DIM = 128
    TIME_DIM = 128
    NUM_EPOCHS = 50
    BATCH_SIZE = 200
    CHECKPOINT_PATH = "data/processed/tgn_checkpoint.pt"
    
    # Initialize Models
    memory = MemoryModule(num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM, memory_dim=MEM_DIM, time_dim=TIME_DIM).to(device)
    gnn = CyberTGN(memory, in_channels=MEM_DIM, out_channels=MEM_DIM, msg_dim=MSG_DIM, time_dim=TIME_DIM).to(device)
    prob_head = AttackProbabilityHead(MEM_DIM).to(device)
    mitre_head = MitreStageClassifier(MEM_DIM, num_classes=14).to(device)
    
    optimizer = Adam([
        {'params': memory.parameters()},
        {'params': gnn.parameters()},
        {'params': prob_head.parameters()},
        {'params': mitre_head.parameters()},
    ], lr=1e-4)

    # Resume from checkpoint if it exists
    start_epoch = load_checkpoints(CHECKPOINT_PATH, memory, gnn, prob_head, mitre_head, optimizer)
    
    # Load Dataset (Assumes data is preprocessed into PyG TemporalData objects)
    data_files = glob.glob("data/processed/temporal_data_*.pt")
    if not data_files:
        print("No processed data found in data/processed/. Please generate temporal_data_X.pt files first.")
        print("Exiting...")
        exit(0)
    
    print(f"Found {len(data_files)} data chunks.")
    
    # For a real pipeline, you might load these sequentially or use a custom PyG Dataset wrapper.
    # Here we simulate loading one chunk for training per epoch as an example.
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        
        for file in data_files:
            data_chunk = torch.load(file)
            # Create a simple dataloader for this chunk
            loader = DataLoader([data_chunk], batch_size=BATCH_SIZE)
            
            # Setup NeighborLoader for this chunk (dummy setup, requires full graph edges typically)
            neighbor_loader = LastNeighborLoader(data_chunk.num_nodes, size=10, device=device)
            
            chunk_loss = train_epoch(loader, memory, gnn, prob_head, mitre_head, neighbor_loader, optimizer, device)
            epoch_loss += chunk_loss
            
        print(f"Epoch {epoch} complete. Loss: {epoch_loss / len(data_files):.4f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'memory': memory.state_dict(),
            'gnn': gnn.state_dict(),
            'prob_head': prob_head.state_dict(),
            'mitre_head': mitre_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")
