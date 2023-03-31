import numpy as np


## coordinates
def cart2sph(x, y, z):
    """Return spherical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """Return Cartesian given Spherical coordinates."""
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def cart2cyl(x, y, z):
    """Return Cylndrical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arcsin(y / r)
    return r, theta, z


def cyl2cart(r, theta, z):
    """Return Cartesian given Cylndrical coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


## normals
# canonical
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


# complex point cloud models
def _pca(X):
    X = X - X.mean(axis=0)
    C = X.T @ X
    U, S, _ = np.linalg.svd(C)
    return U, S


def _estimate_normals_pca(xyz, knn):
    from scipy.spatial import KDTree
    tree = KDTree(xyz)
    n = np.empty_like(xyz)
    for i, query_point in enumerate(xyz):
        nbh_dist, nbh_idx = tree.query([query_point], k=knn)
        query_nbh = xyz[nbh_idx[0]].T
        eigenvec, eigenval = _pca(query_nbh.T)
        n_idx = np.where(eigenval == eigenval.min())[0][0]
        n[i, :] = eigenvec[n_idx]
    return n


def estimate_normals_pca(xyz, take_every=1, knn=30, fast=True):
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
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn),
            fast_normal_computation=fast
        )
        pcd.normalize_normals()
        n = np.asarray(pcd.normals)
    except ImportError as e:
        print('Module `open3d` is not found.\n'
              'Proceeding with own normal estimation algorithm.')
        n = _estimate_normals_pca(xyz, knn)
    return n


def _step_normals_spline(X, x, y, kx=4, ky=4, grid=False):
    from scipy import interpolate
    h = interpolate.SmoothBivariateSpline(*X.T, kx=kx, ky=ky)
    n = np.array([-h(x, y, dx=1, grid=grid),
                  -h(x, y, dy=1, grid=grid),
                  np.ones_like(x)])
    return n


def estimate_normals_spline(xyz, unit=True, knn=30):
    """Return estimated normals for a given point cloud.
        
    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    knn : int, optional
        Number of neighbors for KDTree search.
    unit : bool, optional
        If true, lengths of the normals are 1.

    Returns
    -------
    numpy.ndarray
        The number of rows correspond to the number of rows of a given
        point cloud, and each column corresponds to each component of
        the normal vector.
    """
    from scipy import spatial
    n = np.empty_like(xyz)
    tree = spatial.KDTree(xyz)
    for i, query_point in enumerate(xyz):
        nbh_dist, nbh_idx = tree.query([query_point], k=knn)
        query_nbh = xyz[nbh_idx.flatten()]

        X = query_nbh.copy()
        X_norm = X - X.mean(axis=0)
        p_norm = query_point - X.mean(axis=0)
        U, S, VT = np.linalg.svd(X_norm.T)
        X_trans = X_norm @ U
        p_trans = p_norm @ U
        
        ni = _step_normals_spline(X_trans, p_trans[0], p_trans[1])
        ni = np.dot(U, ni)
        
        if unit:
            ni = np.divide(ni, np.linalg.norm(ni, 2))
        n[i, :] = ni
    return n


## normal orientation
# convex surfaces
def orient_normals_cvx(xyz, n):
    """Orient the normals in the outward direction.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndsarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    center = np.mean(xyz, axis=0)
    for i in range(xyz.shape[0]):
        pi = xyz[i, :] - center
        ni = n[i]
        angle = np.arccos(np.clip(np.dot(ni, pi), -1.0, 1.0))
        if (angle > np.pi/2) or (angle < -np.pi/2):
            n[i] = -ni
    return n


# non-convex surfaces
def _compute_emst(xyz):
    """Compute the symmetric Euclidean minimum spanning graph.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetric Euclidian minimum spanning graph.
    """
    from scipy import spatial
    from scipy import sparse
    tri = spatial.Delaunay(xyz)
    edges = ((0, 1), (1, 2), (0, 2))
    delaunay_edges = None
    for edge in edges:
        if delaunay_edges is None:
            delaunay_edges = tri.simplices[:, edge]
        else:
            delaunay_edges = np.vstack((delaunay_edges,
                                        tri.simplices[:, edge]))
    euclidean_weights = np.linalg.norm((xyz[delaunay_edges[:, 0], :]
                                        - xyz[delaunay_edges[:, 1]]),
                                       axis=1)
    delaunay_euclidean_graph = sparse.csr_matrix((euclidean_weights,
                                                  delaunay_edges.T),
                                                 shape=(xyz.shape[0],
                                                        xyz.shape[0]))
    emst = sparse.csgraph.minimum_spanning_tree(delaunay_euclidean_graph,
                                                overwrite=True)
    return emst + emst.T


def _compute_kgraph(xyz, knn):
    """Compute a graph whose edge (i, j) is nonzero iff j is in the
    k nearest neighborhood of i or i is in the k-neighborhood of j.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndsarray
    knn : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    
    Returns
    -------
    scipy.sparse.coo_matrix
        Symmetric graph.
    """
    from sklearn import neighbors
    kgraph = neighbors.kneighbors_graph(xyz, knn).tocoo()
    return kgraph + kgraph.T


def _compute_rmst(xyz, n, knn, eps=1e-4):
    """Compute the Riemannian minimum spanning tree.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndsarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    knn : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    eps : float, optional
        Value added to the weight of every edge of the Riemannian
        minimum spanning tree.
        
    Returns
    -------
    sparse.csgraph.minimum_spanning_tree
        The Riemannian minimum spanning tree.
    """
    from scipy import sparse
    symmetric_emst = _compute_emst(xyz)
    symmetric_kgraph = _compute_kgraph(xyz, knn)
    enriched = (symmetric_emst + symmetric_kgraph).tocoo()

    conn_l = enriched.row
    conn_r = enriched.col
    riemannian_weights = [1 + eps - np.abs(np.dot(n[conn_l[k],:],
                                                  n[conn_r[k], :]))
                          for k in range(len(conn_l))]
    riemannian_graph = sparse.csr_matrix((riemannian_weights,
                                          (conn_l, conn_r)),
                                         shape=(xyz.shape[0],
                                                xyz.shape[0]))
    rmst = sparse.csgraph.minimum_spanning_tree(riemannian_graph,
                                                overwrite=True)
    return rmst + rmst.T


def _acyclic_graph_iterator(graph, seed):
    """Compute the iterator (depth-first) for an unoriented acyclic
    graph.
    
    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The unoriented acyclic graph.
    seed : int
        The root of the graph.
    
    Returns
    -------
    graph_iterator : iterator
    """
    from scipy import sparse
    from queue import LifoQueue
    graph = sparse.csr_matrix(graph)
    stack = LifoQueue()
    stack.put((None, seed))
    while not stack.empty():
        parent, child = stack.get()
        connected_to_child = graph[child, :].nonzero()[1]
        for second_order_child in connected_to_child:
            if second_order_child != parent:
                stack.put((child, second_order_child))
        yield parent, child
        

def orient_normals(xyz, n, knn):
    """Orient the normals with respect to consistent tangent planes.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
         
    Note: The code is adjusted version of the implementation of
    an algorithm for consistent propagation of normals in unorganized
    set of points in
    https://github.com/PFMassiani/consistent_normals_orientation.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndsarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    knn : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    no = n.copy()
    # compute the Riemannian minimum spanning tree
    symmetric_rmst = _compute_rmst(xyz, n, knn)
    # set the seed and define the orientation
    seed_idx = np.argmax(xyz[:, 2])
    ez = np.array([0, 0, 1])
    if no[seed_idx, :].T @ ez < 0:
        no[seed_idx, :] *= -1
    # traverse the MST (depth first order) to assign a consistent orientation
    for parent_idx, point_idx in _acyclic_graph_iterator(symmetric_rmst,
                                                         seed_idx):
        if parent_idx is None:
            parent_normal = no[seed_idx, :]
        else:
            parent_normal = no[parent_idx, :]

        if no[point_idx, :] @ parent_normal < 0:
            no[point_idx, :] *= -1
    return no


## representation of normals
def n2rgb(n):  # RGB-cube
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
    n = np.divide(n, np.tile(np.expand_dims(
        np.sqrt(np.sum(np.square(n), axis=1)), axis=1),
        [1, 3]
    ))
    rgb = 127.5 + 127.5 * n
    return rgb / 255.0


## curvature
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


## hidden point removal
def remove_hidden_points(xyz, pov, p=np.pi):  # RHP operator
    """Remove points of the point cloud that are not directly visible
    from a preset point of view.
    
    Ref: Katz et al. ACM Transactions on Graphics 26(3), pp: 24-es
         doi: 10.1145/1276377.1276407
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3) where N is the total number of
        points.
    pov : numpy.ndarray
        Point of view for hidden points removal of shape (1, 3).
    p : float
        Parameter for the radius of the spherical transformation.
    
    Returns
    -------
    numpy.ndarray
        Indices of the directly visible points in a point cloud.
    """
    from scipy import spatial
    xyzt = xyz - pov  # move pov to the origin
    norm = np.linalg.norm(xyzt, axis=1)[:, np.newaxis]
    R = norm.max() * 10 ** p
    xyzf = xyzt + 2 * (R - norm) * (xyzt / norm) # perform spherical flip
    hull = spatial.ConvexHull(np.append(xyzf, [[0,0,0]], axis=0))
    return hull.vertices[:-1]


## quadrature functions
def estimate_surface_area(points, normals, bbox=None, n=6, **kwargs):
    """Return the approximate value of the surface area of a non-planar
    surface given tangential points and curvature normals to that
    points.
    
    Parameters
    ----------
    points : numpy.ndarray
        Data points of shape (N, 2), where N is the number of points.
    normals : numpy.ndarray
        Normals.
    n : int, optional
        Non-negative power of 2 to define the number of samples for
        the Romberg quadrature.
    bbox : list, optional
        Bounding box that defines integration domain.
    kwargs : dict, optional
        Additional keyword arguments for
        `scipy.interpolate.SmoothBivariateSpline`.
    
    Returns
    -------
    float
        Approximation of the surface area.
    """
    if not isinstance(normals, np.ndarray):
        raise Exception('`normals` must be array-like.')
    try:
        if not bbox:
            bbox = [points[:, 0].min(), points[:, 0].max(),
                    points[:, 1].min(), points[:, 1].max()]
    except TypeError:
        print('`points` must be a 2-column array.')
    else:
        from scipy import interpolate
        from scipy import integrate
        length = np.linalg.norm(normals, axis=1)
        f = interpolate.SmoothBivariateSpline(*points.T, length, **kwargs)
        num = 2 ** n + 1
        x = np.linspace(bbox[0], bbox[1], num)
        y = np.linspace(bbox[2], bbox[3], num)
        dx = (bbox[1] - bbox[0]) / (num - 1)
        dy = (bbox[3] - bbox[2]) / (num - 1)
        z = f(x, y)
        return integrate.romb(
            integrate.romb(z, dy), dx)


def edblquad(points, values, bbox=None, **kwargs):
    """Return the approximate value of the integral for sampled 2-D
    data by using the spline interpolation.
    
    Parameters
    ----------
    points : numpy.ndarray
        Data points of shape (N, 2), where N is the number of points.
    values : numpy.ndarray
        Sampled integrand function values of shape (N, ).
    bbox : list, optional
        Bounding box that defines integration domain.
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
        from scipy import interpolate
        func = interpolate.SmoothBivariateSpline(*points.T, values, **kwargs)
        return func.integral(*bbox)
