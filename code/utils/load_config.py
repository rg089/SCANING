import json
import os


def load_config():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(curr_path, "..", "config.json")
    with open(config_path, "r") as f:
        dict = json.load(f)
    return dict