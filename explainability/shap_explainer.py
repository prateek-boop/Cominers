import shap
import numpy as np

class FeatureExplainer:
    def __init__(self, model_wrapper_fn, background_data: np.ndarray):
        self.explainer = shap.KernelExplainer(model_wrapper_fn, background_data)
        self.feature_names = [
            'duration', 'packet_count', 'byte_count', 'mean_iat', 'std_iat',
            'syn_count', 'ack_count', 'pkt_size', 'ttl', 'entropy', 'window',
            'syn', 'ack', 'rst', 'fin', 'psh'
        ]
        
    def explain(self, instance: np.ndarray):
        return self.explainer.shap_values(instance)
