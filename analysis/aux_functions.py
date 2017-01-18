import sys
import numpy as np

#=========================================================================


def sort_roots_angle(roots):
    """Radial sort of the roots

    Inspired from:
    http://stackoverflow.com/questions/35606712/numpy-way-to-sort-out-a-messy-array-for-plotting

    """
    # Get one long array with all the roots
    reshaped = roots.reshape((roots.size,), order='F')

    # Separate the real and imaginary parts
    x, y = np.real(reshaped), np.imag(reshaped)

    # Get the angle wrt the mean of the cloud of points
    x0, y0 = x.mean(), y.mean()
    angle = np.arctan2(y - y0, x - x0)

    # Sort based on this angle
    idx = angle.argsort()

    return x[idx], y[idx]


#=========================================================================
def delta(i, j):
    """Given two integers, return 1 if i==j and 0 if not"""

    if isinstance(i, int) and isinstance(i, int):
        if i == j:
            return 1
        else:
            return 0
    else:
        sys.exit('Call the delta function with integers only! Exiting.')
