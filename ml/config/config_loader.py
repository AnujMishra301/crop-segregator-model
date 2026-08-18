"""
Configuration Loader Module
Parses baseline_config.yaml into Python objects to prevent hardcoding parameters across codebase.
"""

import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "baseline_config.yaml")

def load_config(config_path=CONFIG_PATH):
    """Loads configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Configuration Loaded Successfully:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
