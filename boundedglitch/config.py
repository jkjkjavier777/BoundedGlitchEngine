"""Configuration loader."""
import json
from pathlib import Path

def load_config(config_path="bot.json"):
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    print(f"[✓] Configuration loaded from {config_path}")
    return config
