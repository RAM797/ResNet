import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    data_path = os.listdir(data_dir)
    x_train = np.array([[]]).reshape(0,3072)
    y_train = np.array([])
    for df in data_path:
        if df.endswith("html") or df.endswith("meta"):
            continue
        with open(os.path.join(data_dir,df), 'rb') as fo:
            ds = pickle.load(fo, encoding='bytes')
            x_cur = np.array(ds[b'data']) 
            y_cur = np.array(ds[b'labels'])

        if df.startswith("test"):
            x_test = x_cur
            y_test = y_cur

        if df.startswith("data"):
            x_train = np.concatenate((x_cur,x_train), axis=0)
            y_train = np.concatenate((y_cur,y_train), axis=0)


    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
