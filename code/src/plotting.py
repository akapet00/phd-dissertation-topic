import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import contextlib


@contextlib.contextmanager
def set_defense_context():
    import seaborn as sns
    plt.rcParams.update({'text.usetex': True,
                         'text.latex.preamble': r'\usepackage{amsmath}',
                         'font.family': 'serif'})
    sns.set(style='ticks', font='serif', font_scale=1.2,
            palette=None, color_codes=True)
    yield
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def set_axes_equal(ax):
    """Return adjusted axes ratios of a 3-D subplot.

    Note. See https://stackoverflow.com/a/31364297/15005103 for
    implementation details.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D subplot with unequal ratio.

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Subplot with adjusted ratios of axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax


def draw_unit_cube(ax, s=800):
    """Return the RGB unit cube.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D subplot where the RGB unit cube will be drawn.
    s : float, optional
        Size of a marker on a vertex of the cube.

    Returns
    -------
    tuple
        Figure and a 3-D subplot within.
    """
    # create the skeleton of the cube
    r = [0, 1]
    X, Y = np.meshgrid(r, r)
    ones = np.ones(4).reshape(2, 2)
    zeros = np.zeros_like(ones)
    ax.plot_surface(X, Y, zeros, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, Y, zeros, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, zeros, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, ones, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(ones, X, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(zeros, X, Y, lw=2, color='None', edgecolor='k')
    # add colorized points
    pts = np.array(list(itertools.product([0, 1], repeat=3)))
    ax.scatter(*pts.T, c=pts, edgecolor='k', depthshade=False, s=s)
    return ax
