import numpy as np


def get_mae(test_arr, orig_arr) -> float:
    num_data = len(test_arr)
    if num_data != len(orig_arr):
        raise RuntimeError("Two arrays must be of the same size")
    return float(np.mean(abs(test_arr - orig_arr)))


def get_rmse(test_arr, orig_arr) -> float:
    num_data = len(test_arr)
    if num_data != len(orig_arr):
        raise RuntimeError("Two arrays must be of the same size")
    return np.sqrt(np.mean(np.square(test_arr - orig_arr)))
