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

if __name__ == "__main__":
    print("Training script ready for GPU execution.")
