import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

@torch.no_grad()
def test_epoch(loader, memory, gnn, prob_head, mitre_head, neighbor_loader, device):
    memory.eval()
    gnn.eval()
    prob_head.eval()
    mitre_head.eval()
    
    y_true_prob = []
    y_pred_prob = []
    y_true_mitre = []
    y_pred_mitre = []
    
    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        z_mem, last_update = memory(n_id)
        z = gnn(z_mem, last_update, edge_index, batch.t[e_id], batch.msg[e_id])
        
        pos_out = prob_head(z[src], z[pos_dst])
        mitre_out = mitre_head(z[src], z[pos_dst])
        
        y_true_prob.append((batch.y > 0).cpu().numpy())
        y_pred_prob.append(pos_out.cpu().numpy())
        
        y_true_mitre.append(batch.y.cpu().numpy())
        y_pred_mitre.append(mitre_out.argmax(dim=-1).cpu().numpy())
        
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
    y_true_prob = np.concatenate(y_true_prob)
    y_pred_prob = np.concatenate(y_pred_prob)
    y_true_mitre = np.concatenate(y_true_mitre)
    y_pred_mitre = np.concatenate(y_pred_mitre)
    
    auc = roc_auc_score(y_true_prob, y_pred_prob)
    f1 = f1_score(y_true_mitre, y_pred_mitre, average='macro')
    
    return auc, f1

if __name__ == "__main__":
    print("Evaluation script ready.")
