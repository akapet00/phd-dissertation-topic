import numpy as np


def cart_to_sph(x, y, z):
    """Return spherical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph_to_cart(r, theta, phi):
    """Return Cartesian given Spherical coordinates."""
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def cart_to_cyl(x, y, z):
    """Return Cylndrical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arcsin(y / r)
    return r, theta, z


def cyl_to_cart(r, theta, z):
    """Return Cartesian given Cylndrical coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z
