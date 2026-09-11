"""Create synthetic fixtures for testing the training software, never benchmarking."""
import csv
import json
from pathlib import Path
import numpy as np
from model_runtime import FEATURE_NAMES


def make_fixture(root, rows=64):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(12)
    sources = []
    for group,split in enumerate(('train','validation','calibration','test')):
        filename = split+'.csv'
        with (root/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['src_ip','dst_ip','timestamp','label',*FEATURE_NAMES])
            writer.writeheader()
            for i in range(rows):
                attack = i%2
                features = rng.uniform(1,3,16)*(100 if attack else 1)
                writer.writerow(dict(src_ip='192.0.2.1',dst_ip=f'192.0.2.{2+i%3}',
                                     timestamp=1700000000+1000*group+i,label='attack' if attack else 'benign',
                                     **dict(zip(FEATURE_NAMES,features))))
        sources.append(dict(group=split+'-capture',split=split,path=filename,timestamp_kind='end'))
    manifest = dict(labels={'benign':{'attack':0,'stage':0},'attack':{'attack':1,'stage':None}},sources=sources,
                    note='SYNTHETIC SOFTWARE TEST ONLY; not a real accuracy benchmark')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root/'manifest.json'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    args = parser.parse_args()
    print(make_fixture(args.output))
