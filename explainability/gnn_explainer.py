import torch
from torch_geometric.explain import Explainer, GNNExplainer

class GraphExplainer:
    def __init__(self, model: torch.nn.Module):
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='edge',
                return_type='raw',
            ),
        )
        
    def explain(self, x, edge_index, target_edge_index, **kwargs):
        explanation = self.explainer(x, edge_index, target_edge=target_edge_index, **kwargs)
        return explanation.node_mask, explanation.edge_mask
