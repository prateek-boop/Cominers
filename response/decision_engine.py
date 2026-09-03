import yaml

class DecisionEngine:
    def __init__(self, config_path: str = "configs/response_policy.yaml"):
        with open(config_path, 'r') as f:
            self.policy = yaml.safe_load(f)
            
    def evaluate(self, risk_prob: float, mitre_stage: int, conformal_confirmed: bool) -> str:
        if risk_prob < 0.3:
            return "ALLOW"
        elif risk_prob < 0.6:
            return "ALERT"
        elif 0.6 <= risk_prob < 0.85:
            return "ALERT_MONITOR" if conformal_confirmed else "ALERT"
        else:
            if not conformal_confirmed:
                return "ALERT_MONITOR"
            if mitre_stage in [4, 5]:
                return "BLOCK"
            if mitre_stage == 7:
                return "ISOLATE"
            return "BLOCK"
