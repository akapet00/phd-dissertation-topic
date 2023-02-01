import numpy as np


def _estimate_normals_pca(xyz, knn):
    from scipy.spatial import KDTree
    tree = KDTree(xyz)
    n = np.empty_like(xyz)
    for i, query_point in enumerate(xyz):
        nbh_dist, nbh_idx = tree.query([query_point], k=knn)
        query_nbh = xyz[nbh_idx[0]].T 
        X = query_nbh.T.copy()
        X = X - X.mean(axis=0)
        C = X.T @ X
        U, S, VT = np.linalg.svd(C)
        n_idx = np.where(S == S.min())[0][0]
        n[i, :] = U[n_idx]
    return n


def estimate_normals(xyz, take_every=1, knn=30, fast=True):
    """Return estimated normals for a given point cloud.
    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    take_every : int, optional
        How many points to skip in the point cloud when estimating
        normal vectors.
    knn : int, optional
        Number of neighbors for KDTree search.
    fast : bool, optional
        If True, the normal estimation uses a non-iterative method to
        extract the eigenvector from the covariance matrix. This is
        faster, but is not as numerical stable. Only available if
        `open3d` is installed on the system.

    Returns
    -------
    numpy.ndarray
        The number of rows correspond to the number of rows of a given
        point cloud, and each column corresponds to each component of
        the normal vector.
    """
    xyz = xyz[::take_every, :]
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn),
            fast_normal_computation=fast
        )
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(knn)
        n = np.asarray(pcd.normals)
    except ImportError as e:
        print('Module `open3d` is not found.\n'
              'Proceeding with own normal estimation algorithm.')
        n = _estimate_normals_pca(xyz, knn)
    return n


def estimate_curvature(xyz, radius=5):
    """Extract curvature at each point of the point cloud.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud to search for neighbors of.
    radius : numpy.array or float or int, optional
        The radius of points to return.
    
    Returns
    -------
    numpy.ndarray
        A curvature map.
    """
    from scipy.spatial import KDTree
    tree = KDTree(points)
    curvature = [0] * points.shape[0]
    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)
        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V
        curvature[index] = h3 / (h1 + h2 + h3)
    return np.asarray(curvature)


def normals_to_rgb(n):  # RGB-cube
    """Return RGB color representation of unit vectors.
    
    Ref: Ben-Shabat et al., in proceedings of CVPR 2019, pp. 10104-10112,
         doi: 10.1109/CVPR.2019.01035.
    
    Parameters
    ----------
    n : numpy.ndsarray
        The number of rows correspond to the number of rows of a given
        point cloud, each column corresponds to each component of a
        (normalized) normal vector.
    Returns
    -------
    numpy.ndarray
        corresponding RGB color on the RGB cube
    """
    if n.shape[1] != 3:
        raise ValueError('`n` should be a 3-dimensional vector.')
    n = np.divide(
        n, np.tile(
            np.expand_dims(
                np.sqrt(np.sum(np.square(n), axis=1)), axis=1), [1, 3]))
    rgb = 127.5 + 127.5 * n
    return rgb / 255.0
