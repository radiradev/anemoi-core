import numpy as np


def convert_to_timedelta(value: str) -> np.timedelta64:
    """Convert string to timedelta.

    Arguments
    ---------
    value : str
        The timedelta string. Options: 1D, 24h, 6h, 1h, 30m, 10m
    """
    time_res = value[-1]
    num = int(value[:-1])
    return np.timedelta64(num, time_res)
