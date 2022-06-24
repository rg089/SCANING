import os, json


def load_config():
    config_file = os.path.join(os.path.dirname(__file__), 'metrics', 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    return config