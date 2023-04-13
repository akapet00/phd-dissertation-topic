import os

import numpy as np


def load_ear_pcd():
    """Load ear model.

    Parameters
    ----------
    None

    Returns
    -------
    numpy.ndarray
        Point cloud model.
    """
    path = os.path.join('data', 'ear')
    try:
        xyz = np.genfromtxt(f'{path}.xyz', delimiter=',')
        return xyz
    except IOError as e:
        print(e)
        return ''
