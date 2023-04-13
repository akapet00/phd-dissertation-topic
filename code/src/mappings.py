import numpy as np


def encode_normals_rgb(n):
    """Map vectors into corresponding RGB colors considering the RGB cube.

    Note. See
    https://www.mathworks.com/matlabcentral/fileexchange/71178-normal-vector-to-rgb
    for implementation details.

    Parameters
    ----------
    n : numpy.ndsarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.

    Returns
    -------
    numpy.ndarray
        Array of RGB color values (N, 3) for each normal.
    """
    n = np.divide(n, np.tile(np.expand_dims(
        np.sqrt(np.sum(np.square(n), axis=1)), axis=1), [1, 3]))
    rgb = 127.5 + 127.5 * n
    return rgb / 255.0
