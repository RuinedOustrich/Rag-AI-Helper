import yaml
import os


def load_config(config_file):
    """
    Load the configuration from a YAML file.
    """
    if os.path.exists(config_file):
        config = {}
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print("Config file not found, creating default config.yaml...")
        create_config(os.path.join(config_file))
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config


def create_config(config_file, config: dict = None):
    """
    Create a new configuration file.
    """

    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)
