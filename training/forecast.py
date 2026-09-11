"""Train latent delta forecasts from independently captured, closed-window states.

Input manifest binds source NPZ files to an encoder hash and the serving state
schema. Each NPZ has `states` [windows, dimensions] and integer `windows` [windows].
Only train and validation groups are accepted. No test set is read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from forecasting.window_forecaster import NormalizedDeltaPredictor, WindowForecaster, STATE_SCHEMA
from training.data import sha256


def load_sources(manifest):
    path = Path(manifest).resolve()
    meta = json.loads(path.read_text())
    if meta.get('state_schema') != STATE_SCHEMA or meta.get('window_seconds') != 15 or meta.get('flow_batch_size') != 1:
        raise ValueError('Unsupported state/window protocol')
    digest = meta.get('encoder_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Expected SHA256 of the frozen encoder checkpoint')
    groups, hashes, loaded = set(), set(), {'train': [], 'validation': []}
    dimension = None
    for source in meta['sources']:
        split, group = source['split'], source['group']
        if split not in loaded or group in groups:
            raise ValueError('Unique groups in train/validation only are required')
        groups.add(group)
        file = path.parent/source['path']
        digest = sha256(file)
        if digest != source['sha256'] or digest in hashes:
            raise ValueError('Source hash mismatch or duplicate capture')
        hashes.add(digest)
        with np.load(file, allow_pickle=False) as data:
            states, windows = data['states'].copy(), data['windows'].copy()
        if sha256(file) != digest:
            raise ValueError('Source changed while reading')
        if (states.ndim != 2 or not len(states) or states.shape[1] == 0 or
            windows.shape != (len(states),) or not np.issubdtype(windows.dtype, np.integer) or
            not np.isfinite(states).all() or np.any(np.diff(windows) <= 0)):
            raise ValueError('Expected finite states and strictly increasing integer window indices')
        if dimension is None: dimension = states.shape[1]
        if states.shape[1] != dimension: raise ValueError('Latent dimension mismatch')
        anchors = [i for i in range(len(states)-3) if np.all(np.diff(windows[i:i+4]) == 1)]
        if not anchors: raise ValueError('Each capture needs at least four consecutive windows')
        x = states[anchors].astype(np.float32)
        y = np.stack([states[np.array(anchors)+step] for step in (1,2,3)], axis=1).astype(np.float32)
        loaded[split].append((group, x, y, states.astype(np.float32), windows))
    if not all(loaded.values()): raise ValueError('Both training and validation captures required')
    return meta, loaded


def fit(manifest, output, epochs=50, patience=5, seed=42, hidden_dim=128, threads=2):
    if min(epochs, patience, hidden_dim, threads) < 1: raise ValueError('Training sizes must be positive')
    meta, data = load_sources(manifest)
    torch.set_num_threads(threads); torch.manual_seed(seed)
    x = torch.from_numpy(np.concatenate([v[1] for v in data['train']]))
    y = torch.from_numpy(np.concatenate([v[2] for v in data['train']]))
    model = NormalizedDeltaPredictor(x.shape[1], hidden_dim)
    model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0, unbiased=False).clamp_min(.001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    best, stale, history = float('inf'), 0, []
    for epoch in range(epochs):
        model.train(); total = 0.
        for indices in torch.randperm(len(x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            result = model(x[indices])
            predicted = torch.stack([result[f'forecast_{s}s'] for s in (15,30,45)], dim=1)
            loss = (((predicted-y[indices])/model.scale)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite forecast loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(indices)
        model.eval(); captures = {}; weighted_error = 0.; count = 0
        with torch.no_grad():
            for group, vx, vy, observed_states, observed_windows in data['validation']:
                errors = np.zeros(3); base = np.zeros(3)
                for start in range(0,len(vx),128):
                    batch = torch.from_numpy(vx[start:start+128]); target = torch.from_numpy(vy[start:start+128])
                    out = model(batch)
                    pred = torch.stack([out[f'forecast_{s}s'] for s in (15,30,45)], dim=1)
                    errors += (((pred-target)/model.scale)**2).mean(2).sum(0).numpy()
                    base += (((batch[:,None,:]-target)/model.scale)**2).mean(2).sum(0).numpy()
                errors /= len(vx); base /= len(vx)
                # Validate the actual serving recurrence, including Kalman
                # corrections and gap resets, rather than only raw head outputs.
                service = WindowForecaster(x.shape[1], meta['encoder_sha256'])
                service.model = model
                corrected_errors = np.zeros(3)
                anchors = 0
                scale = model.scale.numpy()
                for j, window in enumerate(observed_windows):
                    result = service.observe_window(int(window), observed_states[j], int(window+1)*15)
                    if j+3 >= len(observed_states) or not np.all(np.diff(observed_windows[j:j+4]) == 1):
                        continue
                    forecast = np.array([item['latent_state'] for item in result['forecasts']])
                    corrected_errors += np.mean(((forecast-observed_states[j+1:j+4])/scale)**2, axis=1)
                    anchors += 1
                corrected_errors /= anchors
                captures[group] = dict(samples=len(vx), normalized_mse=corrected_errors.tolist(),
                                       direct_normalized_mse=errors.tolist(), persistence_normalized_mse=base.tolist())
                weighted_error += corrected_errors.mean()*len(vx); count += len(vx)
        score = weighted_error/count
        validation = dict(normalized_mse=float(score), per_capture=captures,
                          scope='Closed-window serving recurrence with Kalman correction and gap resets',
                          beats_persistence_all_horizons=all(all(e < b for e,b in zip(v['normalized_mse'], v['persistence_normalized_mse'])) for v in captures.values()))
        row = dict(epoch=epoch+1, train_normalized_mse=total/len(x), validation=validation)
        history.append(row); (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if score < best:
            best, stale = score, 0
            torch.save(dict(kind='latent_window_forecast_v1', state_schema=STATE_SCHEMA, window_seconds=15,
                            flow_batch_size=1,
                            encoder_sha256=meta['encoder_sha256'], state_dim=x.shape[1], hidden_dim=hidden_dim,
                            state_dict=model.state_dict(), epoch=epoch+1, validation=validation,
                            manifest_sha256=sha256(manifest), sources=meta['sources'], seed=seed,
                            deployment_approved=False), output/'best.pt')
        else: stale += 1
        if stale >= patience: break
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50); parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    fit(**vars(parser.parse_args()))


if __name__ == '__main__': main()
