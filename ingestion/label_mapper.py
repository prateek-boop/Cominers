import yaml
from typing import Dict

class LabelMapper:
    def __init__(self, config_path: str = "data/mappings/mitre_mapping.yaml"):
        self.mapping: Dict[str, int] = {}
        self.mitre_stages = [
            'benign',
            'reconnaissance',
            'initial_access',
            'credential_access',
            'lateral_movement',
            'command_and_control',
            'exfiltration',
            'impact'
        ]
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                for i, stage in enumerate(self.mitre_stages[1:], 1):
                    for label in config.get(stage, []):
                        self.mapping[label.lower()] = i
        except FileNotFoundError:
            pass

    def map_to_int(self, label: str) -> int:
        label = str(label).lower()
        if label == 'benign' or label == 'normal':
            return 0
        return self.mapping.get(label, 0)
