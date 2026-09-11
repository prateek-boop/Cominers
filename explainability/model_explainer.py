"""Model-based explanations of a frozen, pre-update TGN prediction.

Feature attributions are permutation estimates of single-reference interventional
Shapley values. Graph explanations optimize PyG GNNExplainer edge masks. Neither
method explains the heuristic score or establishes that a connection is malicious.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from model_runtime import FEATURE_NAMES


def permutation_shap(predict, features, baseline, permutations=32, seed=42):
    x = np.asarray(features, dtype=np.float32)
    reference = np.asarray(baseline, dtype=np.float32)
    if x.ndim != 1 or reference.shape != x.shape or not np.isfinite([x, reference]).all():
        raise ValueError('Expected matching finite feature vectors')
    if not 2 <= permutations <= 256:
        raise ValueError('permutations must be between 2 and 256')
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(x)) for _ in range(permutations)]
    coalitions = []
    for order in orders:
        point = reference.copy()
        coalitions.append(point.copy())
        for feature in order:
            point[feature] = x[feature]
            coalitions.append(point.copy())
    scores = np.asarray(predict(np.stack(coalitions)), dtype=np.float64)
    if scores.shape != (len(coalitions),) or not np.isfinite(scores).all():
        raise ValueError('Predictor must return one finite score per coalition')
    paths = scores.reshape(permutations, len(x) + 1)
    contributions = np.zeros((permutations, len(x)))
    for i, order in enumerate(orders):
        contributions[i, order] = np.diff(paths[i])
    attribution = contributions.mean(axis=0)
    return dict(values=attribution.tolist(), standard_errors=(contributions.std(axis=0, ddof=1) / np.sqrt(permutations)).tolist(),
                base_value=float(paths[:, 0].mean()), prediction=float(paths[:, -1].mean()),
                additivity_residual=float(paths[0, -1] - paths[0, 0] - attribution.sum()),
                permutations=permutations, seed=seed)


class FrozenPrediction(nn.Module):
    """One flow's attack logit with node memory and history fixed before scoring."""
    def __init__(self, engine, nodes, edges, times, messages, src, dst, current):
        super().__init__()
        self.embedding = deepcopy(engine.gnn.embedding)
        self.head = deepcopy(engine.prob_head)
        self.direct = deepcopy(engine.current_flow_head)
        memory, updated = engine.memory(nodes)
        for name, value in dict(x=memory, updated=updated, edges=edges, times=times,
                                messages=messages, src=src, dst=dst, current=current, nodes=nodes).items():
            self.register_buffer(name, value.detach().clone())
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x, edge_index, flow_index=0):
        z = self.embedding(x, self.updated, edge_index, self.times, self.messages)
        value = self.head(z[self.src[flow_index:flow_index+1]], z[self.dst[flow_index:flow_index+1]])
        if self.direct is not None:
            value = value + self.direct(self.current[flow_index:flow_index+1])[0]
        return value

    def explain_features(self, flow_index, permutations=32):
        with torch.no_grad():
            z = self.embedding(self.x, self.updated, self.edges, self.times, self.messages)
            graph_logit = self.head(z[self.src[flow_index:flow_index+1]], z[self.dst[flow_index:flow_index+1]])
        def predict(values):
            with torch.no_grad():
                logits = graph_logit.expand(len(values))
                if self.direct is not None:
                    logits = logits + self.direct(torch.from_numpy(values))[0]
                return logits.sigmoid().numpy()
        baseline = self.direct.mean.numpy() if self.direct is not None else np.zeros(self.current.shape[1])
        result = permutation_shap(predict, self.current[flow_index].numpy(), baseline, permutations)
        result.update(method='permutation_shapley', explained_output='graph_model_attack_probability',
                      feature_space='signed_log1p_flow_features', baseline=baseline.tolist(),
                      baseline_source='training_feature_mean' if self.direct is not None else 'zero_reference_no_current_flow_branch',
                      scope='Current features only; temporal memory/history held fixed. Feature correlations are not preserved.')
        entries = [dict(feature=name, importance_weight=value, standard_error=error)
                   for name, value, error in zip(FEATURE_NAMES, result['values'], result['standard_errors'])]
        result['attributions'] = entries
        result['top_attributions'] = sorted(entries, key=lambda row: -abs(row['importance_weight']))[:3]
        result['primary_driver'] = result['top_attributions'][0]['feature'] if any(result['values']) else 'No current-feature contribution'
        return result

    def explain_edges(self, flow_index, epochs=50, seed=42, max_edges=2000):
        if not 1 <= epochs <= 500:
            raise ValueError('epochs must be between 1 and 500')
        count = self.edges.shape[1]
        info = dict(method='pyg_gnnexplainer', explained_output='graph_model_attack_logit',
                    scope='Historical edges with pre-update node memory fixed; does not explain how past events created memory.',
                    highlighted_edges=[], vulnerable_edge_count=0)
        if count == 0:
            return dict(info, status='no_historical_edges')
        if count > max_edges:
            return dict(info, status='edge_budget_exceeded', historical_edges=count)
        from torch_geometric.explain import Explainer, GNNExplainer
        # PyG installs masks during optimization; use a private copy, never serving modules.
        model = deepcopy(self)
        with torch.random.fork_rng(devices=[]), torch.enable_grad():
            torch.manual_seed(seed)
            explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=epochs),
                                  explanation_type='model', edge_mask_type='object',
                                  model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'))
            explanation = explainer(model.x, model.edges, flow_index=flow_index)
        weights = explanation.edge_mask.detach().numpy()
        if not np.isfinite(weights).all():
            raise ValueError('Nonfinite explanation mask')
        ranked = np.argsort(-weights)[:3]
        edges = [dict(edge_index=int(i), src_node=int(self.nodes[self.edges[0, i]]),
                      dst_node=int(self.nodes[self.edges[1, i]]), timestamp=int(self.times[i]),
                      importance_weight=float(weights[i])) for i in ranked if weights[i] > 0]
        with torch.no_grad():
            probability = float(self(self.x, self.edges, flow_index).sigmoid()[0])
        return dict(info, status='computed', highlighted_edges=edges, vulnerable_edge_count=len(edges),
                    edge_mask=weights.tolist(), epochs=epochs, seed=seed, probability=probability,
                    explained_class='attack' if probability >= .5 else 'benign')
