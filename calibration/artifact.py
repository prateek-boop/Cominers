"""Persist calibration with checkpoint and serving-protocol identity."""
import json
from pathlib import Path
import numpy as np
from engine.conformal import SplitConformalPredictor
from calibration.conformal import MultiClassSplitConformal
from calibration.ece import compute_ece


def save_calibration(path, encoder_sha256, protocol, benign_scores, stage_labels=None,
                     stage_probabilities=None, alpha=.10, provenance=None):
    if not provenance or not provenance.get('capture_groups') or not provenance.get('source_sha256'):
        raise ValueError('Calibration capture identities and source hash are required')
    if len(encoder_sha256) != 64 or any(c not in '0123456789abcdef' for c in encoder_sha256):
        raise ValueError('Invalid encoder SHA256')
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError('Serving protocol is required')
    binary = SplitConformalPredictor(alpha)
    binary.calibrate(np.asarray(benign_scores))
    stage = MultiClassSplitConformal(alpha)
    labels = np.empty(0, dtype=np.int64)
    probabilities = np.empty((0, 0), dtype=np.float64)
    diagnostics = dict(stage_calibrated=False, stage_support=[], stage_confidence_ece=None)
    if stage_labels is not None or stage_probabilities is not None:
        raw_labels = np.asarray(stage_labels)
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError('Stage labels must be explicit integer class IDs')
        labels = raw_labels.astype(np.int64)
        probabilities = np.asarray(stage_probabilities, dtype=np.float64)
        stage.calibrate(labels, probabilities)
        diagnostics.update(stage_calibrated=True,
                           stage_support=np.bincount(labels, minlength=probabilities.shape[1]).tolist(),
                           stage_confidence_ece=compute_ece((probabilities.argmax(1)==labels).astype(int), probabilities.max(1)))
    metadata = dict(kind='stream_calibration_v1', encoder_sha256=encoder_sha256, protocol=protocol,
                    alpha=alpha, provenance=provenance, diagnostics=diagnostics,
                    note='Calibration diagnostics are in-sample; they do not validate detection or deployment.')
    # Refuse to silently replace prior calibration evidence.
    with Path(path).open('xb') as output:
        np.savez_compressed(output, metadata=np.array(json.dumps(metadata)),
                            benign_scores=binary.benign_scores, stage_labels=labels,
                            stage_probabilities=probabilities)
    return metadata


def load_calibration(path, encoder_sha256, protocol, num_classes, alpha):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if (meta.get('kind') != 'stream_calibration_v1' or meta.get('encoder_sha256') != encoder_sha256
            or meta.get('protocol') != protocol or meta.get('alpha') != alpha):
            raise ValueError('Calibration does not match checkpoint, protocol, or alpha')
        binary = SplitConformalPredictor(alpha)
        binary.calibrate(data['benign_scores'])
        stage = MultiClassSplitConformal(alpha)
        labels, probs = data['stage_labels'], data['stage_probabilities']
        if labels.size:
            if probs.ndim != 2 or probs.shape[1] != num_classes:
                raise ValueError('Calibration stage dimension does not match checkpoint')
            stage.calibrate(labels, probs)
    return binary, stage, meta
