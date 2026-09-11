"""Matched validation-only comparisons; never evaluate the test set."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    arms = [('cic2017', 'cic2017', False), ('combined', 'combined', False),
            ('family', 'combined', True)]
    summary = []
    for name, source, weighted in arms:
        output = ROOT / f'training-runs/controlled-{name}'
        command = [sys.executable, '-u', '-m', 'training.tabular', 'train',
                   '--data', 'data/ids2018-tabular', '--graph-data', 'data/cic2017-prepared',
                   '--output', str(output), '--training-source', source,
                   '--validation-only', '--epochs', '10', '--patience', '3',
                   '--hidden', '64', '--threads', '2', '--seed', '42']
        if weighted:
            command.append('--family-weighting')
        print(json.dumps(dict(event='starting', arm=name, command=command)), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        summary.append(dict(arm=name, validation=json.loads((output/'validation.json').read_text()),
                            fixed_threshold_diagnostic=json.loads((output/'validation-at-0.5.json').read_text()),
                            history=json.loads((output/'history.json').read_text())))
        (ROOT/'reports/audit/controlled-baselines.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(event='completed', test_evaluated=False)), flush=True)


if __name__ == '__main__':
    main()
