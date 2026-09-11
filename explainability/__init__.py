"""Model-based explanations of frozen TGN predictions."""
from .model_explainer import FrozenPrediction, permutation_shap

__all__ = ["FrozenPrediction", "permutation_shap"]
