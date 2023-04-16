import numpy as np
from scipy import interpolate
from scipy import spatial


def pca(X):
    """Return principal components.
    
    Parameters
    ----------
    X : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    
    Returns
    -------
    tuple
        Eigenvectors and eigenvalues of a covariance matrix of `X`.
    """
    X = X - X.mean(axis=0)
    C = (X.T @ X) / X.shape[0]
    U, S, _ = np.linalg.svd(C)
    return U, S


def remove_hidden_points(xyz, pov, p=np.pi):
    """Return only the points of a given point cloud that are directly
    visible from a preset point of view.
    
    Ref: Katz et al. ACM Transactions on Graphics 26(3), pp: 24-es
         doi: 10.1145/1276377.1276407
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    pov : numpy.ndarray
        Point of view for hidden points removal of shape (1, 3).
    p : float
        Parameter for the radius of the spherical transformation.
    
    Returns
    -------
    numpy.ndarray
        Indices of the directly visible points in a point cloud.
    """
    xyzt = xyz - pov  # move pov to the origin
    norm = np.linalg.norm(xyzt, axis=1)[:, np.newaxis]
    R = norm.max() * 10 ** p
    xyzf = xyzt + 2 * (R - norm) * (xyzt / norm) # perform spherical flip
    hull = spatial.ConvexHull(np.append(xyzf, [[0,0,0]], axis=0))
    return hull.vertices[:-1]


def apply_weight(p, nbhd, kernel='linear', gamma=None):
    """Return scaling weights given distance between a targeted point
    and its surrounding local neighborhood.
    
    Parameters
    ----------
    p : numpy_ndarray
        Targeted point of shape (3, ).
    nbhd : numpy.ndarray
        An array of shape (N, 3) representing the local neighborhood.
    kernel : str, optional
        The weighting function to use for MLS fitting. If not set, all
        weights will be set to 1.
    gamma : float, optional
        A scaling factor for the weighting function. If not given, it
        is set to 1.
        
    Returns
    -------
    numpy.ndarray
        Array with weights of (N, ).
    """
    dist = np.linalg.norm(nbhd - p, axis=1)  # squared Euclidian distance
    if gamma is None:
        gamma = 1.
    if kernel == 'linear':
        w = np.maximum(1 - gamma * dist, 0)
    elif kernel == 'truncated':
        w = np.maximum(1 - gamma * dist ** 2, 0)
    elif kernel == 'inverse':
        w = 1 / (dist + 1e12) ** gamma
    elif kernel == 'gaussian':
        w = np.exp(-(gamma * dist) ** 2)
    elif kernel == 'multiquadric':
        w = np.sqrt(1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_quadric':
        w = 1 / (1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_multiquadric':
        w = 1 / np.sqrt(1 + (gamma * dist) ** 2 )
    elif kernel == 'thin_plate_spline':
        w = dist ** 2 * np.log(dist)
    elif kernel == 'rbf':
        w = np.exp(-dist ** 2 / (2 * gamma ** 2))
    elif kernel == 'cosine':
        w = (nbhd @ p) / np.linalg.norm(nbhd * p, axis=1)
    return w


def polyfit2d(x, y, z, deg=1, rcond=None, full_output=False):
    r"""Return the coefficients of a 2-D polynomial of a given degree.
    This function is the 2-D adapted version of `numpy.polyfit`.
    
    Note. The fitting assumes that the variable `z` corresponds the
    values of the "height" function such as z = f(x, y). The fitting is
    then performed on the polynomial given as:
    
    .. math:: f(x, y) = \sum_{i, j} c_{i, j} x^i y^j
    
    where `i + j Y= n` and `n` is the degree of a polynomial.
    
    Parameters
    ----------
    x, y : array_like, shape (N,)
        x- and y-oordinates of the M data points `(x[i], y[i])`.
    z : array_like, shape (M,)
        z-coordinates of the M data points.
    deg : int, optional
        Degree of the polynomial to be fit.
    rcond : float, optional
        Condition of the fit. See `scipy.linalg.lstsq` for details. All
        singular values less than `rcond` will be ignored.
    full_output : bool, optional
        Full diagnostic information from the SVD is returned if True,
        otherwise only the fitted coefficients are returned.
        
    Returns
    -------
    numpy.ndarray
        Array of coefficients of shape (deg+1, deg+1). If `full_output`
        is set to true, sum of the squared residuals of the fit, the
        effective rank of the design matrix, its singular values, and
        the specified value of `rcond` are also returned.
    """
    deg = int(deg)
    if deg < 1:
        raise ValueError('Degree must be at least 1.')
    # set up the Vandermode (design) matrix and the intercept vector
    A = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg])
    b = z.flatten()
    # set up relative condition of the fit
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps
    # solve the least square
    coef, res, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    if full_output:
        return coef.reshape(deg+1, deg+1), res, rank, s
    return coef.reshape(deg+1, deg+1)


def edblquad(points, values, bbox=None, method=None, **kwargs):
    """Return the approximate solution to the double integral by
    observing sampled integrand function.
    
    Parameters
    ----------
    points : numpy.ndarray
        The point cloud of shape (N, 2), N is the number of points.
    values : numpy.ndarray
        Sampled integrand of shape (N, 3).
    bbox : list, optional
        Bounding box that defines integration domain.
    method : string, optional
        If None, the integral is computed by directly integrating
        splines. Alternative method is `gauss` which utilizes adaptive
        Gauss-Kronrad quadrature.
    kwargs : dict, optional
        Additional keyword arguments for
        `scipy.interpolate.SmoothBivariateSpline`.
    
    Returns
    -------
    float
        Approximation of the double integral.
    """
    if not isinstance(values, np.ndarray):
        raise Exception('`values` must be array-like.')
    try:
        if not bbox:
            bbox = [points[:, 0].min(), points[:, 0].max(),
                    points[:, 1].min(), points[:, 1].max()]
    except TypeError:
        print('`points` must be a 2-column array.')
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = interpolate.SmoothBivariateSpline(*points.T,
                                                  values,
                                                  bbox=bbox,
                                                  **kwargs)
        if method is None:  # default settings
            return f.integral(*bbox)
        if method == 'gauss':
            from scipy import integrate
            f_wrap = lambda v, u: f(u, v)
            I, _ = integrate.dblquad(f, *bbox)
            return I
        else:  
            raise ValueError('Method is not supported')
        return I

    
def estimate_surface_area(xyz, n=None, full_output=False):
    """Return estimated surface area from a smooth representation of a
    point cloud.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud distributed in 3-D space of shape (N, 3) where N is
        the total number of points.
    n : numpy.ndarray, optional
        Oriented normals in 3-D space of shape (N, 3) where N is the
        number of points in a point cloud. If not provided, normals are
        automatically inferred from `xyz` which can lead to numerical
        artifacts depending on the size of the `xyz`.
    full_output : bool, optional
        If True, reconstructed mesh is also returned.
    
    Returns
    -------
    float or tuple
        Estimated surface area and reconstructed mesh (vertices and
        triangles) if `full_output` is True.
    """
    try:
        import open3d as o3d
    except ImportError as e:
        print('Package open3d is required.')
    else:
        N = xyz.shape[0]
        if N < 10:
            raise ValueError('Number of points must be > 10.')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if n is not None:
            pcd.normals = o3d.utility.Vector3dVector(n)
        else:
            knn = int(2 * np.log(N))
            if knn < 5:
                knn = 5
            elif knn > 30:
                knn = 30
            else:
                pass  # keep knn as computed
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
            pcd.orient_normals_consistent_tangent_plane(knn)
        # set the magnitude of each normal to 1
        pcd.normalize_normals()
        # reconstruct triangular mesh
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=6)
        mesh.remove_duplicated_vertices()
        # export reconstructed vertices
        vert = np.asarray(mesh.vertices)
        # set the point cloud in its orthonormal basis
        basis = xyz.copy()
        mu = basis.mean(axis=0)
        basis = basis - mu
        cov = basis.T @ basis
        evec, _, _ = np.linalg.svd(cov)
        basis = basis @ evec
        # create the convex hull in 2-D space
        hull = spatial.Delaunay(basis[:, :2])
        # remove vertices out of the convex hull
        vertt = (vert - mu) @ evec
        vert_mask = hull.find_simplex(vertt[:, :2]) < 0
        mesh.remove_vertices_by_mask(vert_mask)
        # compute the surface area
        area = mesh.get_surface_area()
        if full_output:
            vert = np.asarray(mesh.vertices)
            tri = np.asarray(mesh.triangles)
            return area, vert, tri
        return area
