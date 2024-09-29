import numpy as np


def get_mae(test_arr, orig_arr):
    num_data = len(test_arr)
    if num_data != len(orig_arr):
        raise RuntimeError("Two arrays must be of the same size")
    return abs(test_arr - orig_arr).sum() / num_data


def get_rmse(test_arr, orig_arr):
    num_data = len(test_arr)
    if num_data != len(orig_arr):
        raise RuntimeError("Two arrays must be of the same size")
    return (np.sum(abs(test_arr - orig_arr) ** 2) / num_data) ** 0.5
