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


def create_config(config_file, config=None):
    """
    Create a new configuration file.
    """
    if config is None:
        config = {}
        config['database'] = {}
        config['database']['database_type'] = 'faiss'
        config['database']['database_path'] = './database'
        config['database']['remove_docstr'] = True
        config['database']['chunk_size'] = 1000
        config['database']['chunk_overlap'] = 0
        config['database']['extension'] = 'py'

    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)
