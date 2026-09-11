"""Export closed-window latent states from real PCAPs using the serving pipeline.

Manifest: {"sources": [{"path": "capture.pcap", "sha256": "...",
"group": "capture-name", "split": "train"}]} with separate validation captures.
Source hashes and capture identities are retained; no synthetic traffic or labels
are substituted. Final partial windows are excluded.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from pipeline import CyberDefensePipeline
from forecasting.window_forecaster import STATE_SCHEMA
from training.data import sha256


def export(manifest, checkpoint, output, threads=2):
    if threads < 1: raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    manifest = Path(manifest).resolve()
    sources = json.loads(manifest.read_text())['sources']
    groups = [s['group'] for s in sources]
    if len(set(groups)) != len(groups) or {s['split'] for s in sources} != {'train','validation'}:
        raise ValueError('Unique train and validation captures are required; no test input')
    paths=[];hashes=set()
    for source in sources:
        path=manifest.parent/source['path']; digest=sha256(path)
        if digest != source['sha256'] or digest in hashes:
            raise ValueError('Source mismatch or duplicate capture')
        hashes.add(digest);paths.append(path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    encoder_hash=sha256(checkpoint)
    result=dict(state_schema=STATE_SCHEMA,window_seconds=15,flow_batch_size=1,
                encoder_sha256=encoder_hash,input_manifest_sha256=sha256(manifest),sources=[])
    for i,(source,path) in enumerate(zip(sources,paths)):
        states=[];windows=[]
        with tempfile.TemporaryDirectory(prefix='cybertgn-latent-export-') as tmp:
            def on_window(index,state,observed_at):
                windows.append(index);states.append(state)
            pipeline=CyberDefensePipeline(checkpoint_path=checkpoint,explanations_per_batch=0,
                                          enable_incidents=False,window_observer=on_window,
                                          forensics_dir=Path(tmp)/'pcaps',ledger_file=Path(tmp)/'ledger.jsonl')
            last=None;packets=0
            for packet in pipeline.sniffer.read_pcap(str(path)):
                pipeline.ingest_packet(packet);last=packet.timestamp;packets+=1
            pipeline.flush_and_process()
            if last is not None:pipeline.advance_watermark(last)
        if sha256(path)!=source['sha256'] or sha256(checkpoint)!=encoder_hash:
            raise ValueError('Capture or checkpoint changed during export')
        if not states:raise ValueError(f'No complete observed windows in {source["group"]}')
        destination=output/f'{i:03d}.npz'
        np.savez_compressed(destination,states=np.stack(states),windows=np.array(windows,dtype=np.int64))
        entry=dict(path=destination.name,sha256=sha256(destination),group=source['group'],split=source['split'],
                   pcap_sha256=source['sha256'],observed_windows=len(windows),packets=packets)
        result['sources'].append(entry)
        print(json.dumps(entry),flush=True)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--threads',type=int,default=2)
    export(**vars(parser.parse_args()))


if __name__=='__main__':main()
