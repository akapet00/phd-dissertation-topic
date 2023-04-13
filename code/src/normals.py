import numpy as np
from scipy import interpolate
from scipy import spatial

from .utils import pca, apply_weight, polyfit2d


def sph_normals(r, theta, phi):
    """Return unit vector field components normal to spherical
    surface."""
    nx = r ** 2 * np.cos(phi) * np.sin(theta) ** 2 
    ny = r ** 2 * np.sin(phi) * np.sin(theta) ** 2
    nz = r ** 2 * np.cos(theta) * np.sin(theta)
    return nx, ny, nz


def cyl_normals(r, theta, z):
    """Return unit vector field components normal to cylindrical
    surface."""
    nx = np.cos(theta)
    ny = np.sin(theta)
    nz = np.zeros_like(z)
    return nx, ny, nz


def _estim_normals_pca(xyz, k):
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        # extract the local neighborhood
        _, ind = tree.query([p], k=k)
        nbh = xyz[ind[0]]
        
        # extract an eigenvector with smallest associated eigenvalue
        U, S = pca(nbh)
        n[i, :] = U[:, np.argmin(S)]
    return n


def estim_normals_pca(xyz, k=30, fast=True):
    """Return estimated normals for a given point cloud.

    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    fast : bool, optional
        If True, the normal estimation uses a non-iterative method to
        extract the eigenvector from the covariance matrix. This is
        faster, but is not as numerical stable. Only available if
        `open3d` is installed on the system.

    Returns
    -------
    numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    try:
        import open3d as o3d
    except ImportError as e:
        print('Module `open3d` is not found.'
              'Proceeding with own normal estimation algorithm.',
              sep='\n')
        n = _estim_normals_pca(xyz, k)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(k),
            fast_normal_computation=fast)
        pcd.normalize_normals()
        n = np.asarray(pcd.normals)
    return n


def estim_normals_spline(xyz,
                         k,
                         deg=3,
                         s=None,
                         unit=True,
                         kernel=None, 
                         **kwargs):
    """Return the (unit) normals by constructing smooth bivariate
    B-spline at each point in the point cloud considering its local
    neighborhood.

    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    deg : float, optional
        Degrees of the bivariate spline.
    s : float, optional
        Positive smoothing factor defined for smooth bivariate spline
        approximation.
    unit : float, optional
        If true, normals are normalized. Otherwise, surface normals are
        returned.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `apply_weight` function.

    Returns
    -------
    numpy.ndarray
        The (unit) normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]

        # change the basis of the local neighborhood
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        C = (X.T @ X) / (nbhd.shape[0] - 1)
        U, _, _ = np.linalg.svd(C)
        Xt = X @ U

        # compute weights given specific distance function
        if kernel:
            w = apply_weight(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))

        # create a smooth B-Spline representation of the "height" function
        h = interpolate.SmoothBivariateSpline(*Xt.T, w=w, kx=deg, ky=deg, s=s)

        # compute normals as partial derivatives of the "height" function
        ni = np.array([-h(*Xt[0, :2], dx=1).item(),
                       -h(*Xt[0, :2], dy=1).item(),
                       1])

        # convert normal coordinates into the original coordinate frame
        ni = U @ ni
        
        # normalize normals by considering the magnitude of each
        if unit:
            ni = ni / np.linalg.norm(ni, 2)
        n[i, :] = ni
    return n


def estim_normals_poly(xyz,
                       k,
                       deg=1,
                       unit=True,
                       kernel=None,
                       **kwargs):
    """Return the (unit) normals by fitting 2-D polynomial at each
    point in the point cloud considering its local neighborhood.

    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    deg : float, optional
        Degrees of the polynomial.
    unit : float, optional
        If true, normals are normalized. Otherwise, surface normals are
        returned.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `apply_weight` function.

    Returns
    -------
    numpy.ndarray
        The (unit) normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    n = np.empty_like(xyz)
    tree = spatial.KDTree(xyz)
    for i, p in enumerate(xyz):
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]

        # change the basis of the local neighborhood
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        C = (X.T @ X) / (nbhd.shape[0] - 1)
        U, _, _ = np.linalg.svd(C)
        X_t = X @ U

        # compute weights given specific distance function
        if kernel:
            w = apply_weight(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))

        # fit parametric surface by usign a (weighted) 2-D polynomial
        X_t_w = X_t * w[:, np.newaxis]
        c = polyfit2d(*X_t_w.T, deg=deg)

        # compute normals as partial derivatives of the "height" function
        cu = np.polynomial.polynomial.polyder(c, axis=0)
        cv = np.polynomial.polynomial.polyder(c, axis=1)
        ni = np.array([-np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cu),
                       -np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cv),
                       1])

        # convert normal coordinates into the original coordinate frame
        ni = U @ ni

        # normalize normals by considering the magnitude of each
        if unit:
            ni = ni / np.linalg.norm(ni, 2)
        n[i, :] = ni
    return n
