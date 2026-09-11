"""Strict CSV ingestion, disk-backed chronological sorting and leakage checks."""
import csv
from datetime import datetime
import hashlib
from ipaddress import ip_address
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
from model_runtime import FEATURE_NAMES

SPLITS = ('train', 'validation', 'calibration', 'test')
ALIASES = [
    ['Flow Duration', 'FlowDuration'], ['Total Fwd Packet', 'Total Fwd Packets', 'Tot Fwd Pkts'],
    ['Total Bwd packets', 'Total Backward Packets', 'Tot Bwd Pkts'],
    ['Total Length of Fwd Packet', 'Total Length of Fwd Packets', 'TotLen Fwd Pkts'],
    ['Total Length of Bwd Packet', 'Total Length of Bwd Packets', 'TotLen Bwd Pkts'],
    ['Fwd Packet Length Mean', 'Fwd Pkt Len Mean'], ['Bwd Packet Length Mean', 'Bwd Pkt Len Mean'],
    ['Flow Bytes/s', 'Flow Byts/s'], ['Flow Packets/s', 'Flow Pkts/s'], ['Flow IAT Mean'],
    ['Fwd IAT Mean'], ['Bwd IAT Mean'], ['Fwd Packets/s', 'Fwd Pkts/s'],
    ['Bwd Packets/s', 'Bwd Pkts/s'], ['Average Packet Size', 'Pkt Size Avg'], ['Down/Up Ratio'],
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def resolve_columns(headers, explicit):
    indexed = {}
    for header in headers:
        key = normalize(header)
        if key in indexed:
            raise ValueError(f'Ambiguous duplicate CSV column: {header}')
        indexed[key] = header
    aliases = dict(zip(FEATURE_NAMES, ALIASES))
    aliases.update(src_ip=['src_ip', 'Source IP', 'Src IP'], dst_ip=['dst_ip', 'Destination IP', 'Dst IP'],
                   timestamp=['timestamp'], label=['label'])
    if set(explicit) - set(aliases):
        raise ValueError(f'Unknown column overrides: {sorted(set(explicit) - set(aliases))}')
    result = {}
    for target, choices in aliases.items():
        if target in explicit:
            choices = [explicit[target]]
        matches = {indexed[normalize(c)] for c in choices if normalize(c) in indexed}
        if len(matches) != 1:
            raise ValueError(f'CSV requires an unambiguous {target!r} column. Missing features/IPs cannot be fabricated.')
        result[target] = matches.pop()
    return result


def prepare(manifest_path, output, shard_size=100000):
    """Every manifest entry is one complete capture group in exactly one split."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    config = json.loads(manifest_path.read_text())
    sources, labels = config['sources'], config['labels']
    if shard_size < 1 or not sources:
        raise ValueError('Positive shard size and nonempty sources required')
    groups = [s['group'] for s in sources]
    if len(groups) != len(set(groups)) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('Each capture group must occur exactly once; combine its CSV pieces first')
    if {s['split'] for s in sources} != set(SPLITS):
        raise ValueError(f'Provide disjoint capture groups for all four splits: {SPLITS}')
    for label, spec in labels.items():
        if spec['attack'] not in (0, 1) or spec.get('stage') not in (None, *range(8)):
            raise ValueError(f'Invalid explicit label mapping: {label}')
        if spec.get('stage') == 0 and spec['attack'] != 0:
            raise ValueError('Attack labels cannot map to benign stage 0')
        if spec['attack'] == 0 and spec.get('stage') not in (0, None):
            raise ValueError('Benign labels cannot map to attack stages')
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output / 'ingestion.sqlite')
    db.execute('CREATE TABLE seen (fingerprint TEXT PRIMARY KEY, split TEXT, label INTEGER, stage INTEGER)')
    db.execute('CREATE TABLE rows (timestamp REAL, src INTEGER, dst INTEGER, msg BLOB, label INTEGER, stage INTEGER)')
    metadata = {'version':2, 'feature_names':FEATURE_NAMES, 'manifest':config,
                'manifest_sha256':sha256(manifest_path), 'shards':[], 'sources':[], 'max_nodes':0}
    count = 0
    mean, m2 = np.zeros(16), np.zeros(16)
    source_hashes = set()
    try:
        for source_index, source in enumerate(sources):
            path = (manifest_path.parent / source['path']).resolve()
            digest = sha256(path)
            if digest in source_hashes:
                raise ValueError('Identical source file included more than once')
            source_hashes.add(digest)
            db.execute('DELETE FROM rows')
            mapping = {}
            def node(ip):
                if ip not in mapping:
                    mapping[ip] = len(mapping)
                return mapping[ip]
            duplicates, inserted = 0, 0
            with path.open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream)
                columns = resolve_columns(reader.fieldnames or [], source.get('columns', {}))
                for line, row in enumerate(reader, 2):
                    try:
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError('Malformed CSV row: number of fields differs from header')
                        label_name = row[columns['label']].strip()
                        if label_name not in labels:
                            raise ValueError(f'Unknown label {label_name!r}; add an explicit mapping')
                        label = int(labels[label_name]['attack'])
                        stage = labels[label_name].get('stage')
                        stage = -1 if stage is None else int(stage)
                        src, dst = (str(ip_address(row[columns[k]].strip())) for k in ('src_ip','dst_ip'))
                        features = np.asarray([float(row[columns[n]]) for n in FEATURE_NAMES], dtype=np.float64)
                        if not np.isfinite(features).all() or np.any(features < 0):
                            raise ValueError('Features must be finite and nonnegative; audit/re-extract invalid source rows')
                        time_text = row[columns['timestamp']].strip()
                        if source.get('timestamp_format'):
                            dt = datetime.strptime(time_text, source['timestamp_format'])
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=ZoneInfo(source['timezone']))
                            timestamp = dt.timestamp()
                        else:
                            timestamp = float(time_text)
                        if source.get('timestamp_kind', 'end') == 'start':
                            timestamp += features[0] / 1e6
                        elif source.get('timestamp_kind', 'end') != 'end':
                            raise ValueError('timestamp_kind must be start or end')
                        if not np.isfinite(timestamp) or not 0 <= timestamp <= 1e12:
                            raise ValueError('Timestamp outside supported range')
                        fingerprint = hashlib.sha256(json.dumps([src,dst,timestamp,features.tolist()],separators=(',',':')).encode()).hexdigest()
                        previous = db.execute('SELECT split,label,stage FROM seen WHERE fingerprint=?',(fingerprint,)).fetchone()
                        if previous:
                            if previous != (source['split'], label, stage):
                                raise ValueError('Duplicate event crosses splits or has contradictory labels')
                            duplicates += 1
                            continue
                        db.execute('INSERT INTO seen VALUES (?,?,?,?)',(fingerprint,source['split'],label,stage))
                        msg = np.log1p(features)
                        if np.any(msg > 100):
                            raise ValueError('Transformed features exceed runtime range')
                        db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?)',
                                   (timestamp,node(src),node(dst),msg.astype('<f4').tobytes(),label,stage))
                        inserted += 1
                        if source['split'] == 'train':
                            count += 1
                            delta = msg - mean
                            mean += delta / count
                            m2 += delta * (msg - mean)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f'{path}:{line}: {exc}') from exc
            if not inserted:
                raise ValueError(f'Capture has no unique valid rows: {path}')
            db.commit()
            cursor = db.execute('SELECT timestamp,src,dst,msg,label,stage FROM rows ORDER BY timestamp,rowid')
            shard_index = 0
            while rows := cursor.fetchmany(shard_size):
                name = f'{source_index:04d}-{shard_index:05d}.npz'
                np.savez(output/name, timestamp=np.array([r[0] for r in rows]),
                         src=np.array([r[1] for r in rows],dtype=np.int64), dst=np.array([r[2] for r in rows],dtype=np.int64),
                         msg=np.stack([np.frombuffer(r[3],dtype='<f4') for r in rows]),
                         label=np.array([r[4] for r in rows],dtype=np.int64), stage=np.array([r[5] for r in rows],dtype=np.int64))
                metadata['shards'].append({'path':name,'sha256':sha256(output/name),'group':source['group'],
                                           'split':source['split'],'rows':len(rows)})
                shard_index += 1
            metadata['sources'].append({'path':str(path),'sha256':digest,'rows':inserted,'duplicates_removed':duplicates,
                                        'group':source['group'],'split':source['split'],'nodes':len(mapping)})
            metadata['max_nodes'] = max(metadata['max_nodes'], len(mapping))
        metadata['train_mean'] = mean.tolist()
        metadata['train_scale'] = np.maximum(np.sqrt(m2 / max(count,1)),1e-3).tolist()
        (output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        db.close()
        # Intermediate database belongs only to this preparation run.
        (output/'ingestion.sqlite').unlink(missing_ok=True)
    return metadata


def read_metadata(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'dataset.json').read_text())
    if metadata.get('version') != 2 or metadata['feature_names'] != FEATURE_NAMES:
        raise ValueError('Unsupported prepared dataset schema')
    return metadata


def shards(directory, split, verify=True):
    directory = Path(directory)
    for shard in read_metadata(directory)['shards']:
        if shard['split'] == split:
            path = directory / shard['path']
            if verify and sha256(path) != shard['sha256']:
                raise ValueError(f'Prepared shard hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as data:
                yield shard['group'], {key:data[key] for key in data.files}
