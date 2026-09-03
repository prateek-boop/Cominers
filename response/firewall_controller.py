import subprocess
import logging

class FirewallController:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.logger = logging.getLogger("FirewallController")
        
    def _execute(self, cmd: list) -> bool:
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True
            
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Firewall command failed: {e.stderr}")
            return False

    def block_ip(self, ip: str) -> bool:
        cmd = ["nft", "add", "rule", "inet", "filter", "input", "ip", "saddr", ip, "drop"]
        return self._execute(cmd)

    def isolate_host(self) -> bool:
        cmd = ["nft", "add", "rule", "inet", "filter", "input", "drop"]
        return self._execute(cmd)
