import json
import os
import random


def load_corruptions(fname='corruptions.json', train=True):
    """
    Loads the corruptions from the json file
    :param train: True if training, False if testing
    :return: list of corruptions
    """
    # Path to the current folder
    folder_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(folder_path, 'helper', fname)
    with open(json_path, 'r') as f:
        data = json.load(f)
    key = "train" if train else "inference"
    info = data[key]
    corruption_combs, probabilites = info["corruption_combinations"], info["probabilities"]
    return corruption_combs, probabilites


def jitter(frac, distance=0.15):
    # Jitter the fraction randomly by the given distance
    jittered = frac + distance * (random.random()*2 - 1)
    return max(0, min(1, jittered)) # Make sure the jittered value is between 0 and 1
