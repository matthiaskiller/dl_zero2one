import pickle
import os

import numpy as np


def save_pickle(data_dict, file_name):
    """Save given data dict to pickle file file_name in models/"""
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(data_dict, open(os.path.join(directory, file_name), 'wb', 5))


def load_pickle(file_name):
    """Load data from pickle file"""
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with (open(os.path.join(directory, file_name), "rb")) as openfile:
        return pickle.load(openfile)
