"""Point mass module.

This module defines the PointMass class which will be used to link elements.
"""
import numpy as np
from ross.element import Element

__all__ = ["PointMass"]


class PointMass(Element):
    """A point mass element.

    This class will create a point mass element.
    This element can be used to link other elements in the analysis.
    The mass provided to the element can be different on the x and y direction
    (e.g. different support inertia for x and y directions).

    Parameters
    ----------
    n: int
        Node which the bearing will be located in.
    m: float, optional
        Mass for the element.
    mx: float, optional
        Mass for the element on the x direction.
    my: float, optional
        Mass for the element on the y direction.

    Examples
    --------
    >>> p0 = PointMass(n=0, m=2)
    >>> p0.M()
    array([[2, 0],
           [0, 2]])
    >>> p1 = PointMass(n=0, mx=2, my=3)
    >>> p1.M()
    array([[2, 0],
           [0, 3]])
    """

    def __init__(self, n=None, m=None, mx=None, my=None, tag=None):
        self.n = n
        self.m = m

        if mx is None and my is None:
            mx = m
            my = m

        self.mx = mx
        self.my = my

        if tag is None:
            tag = self.__class__.__name__ + " " + str(self.n)
        self.tag = tag

    def __hash__(self):
        return hash(self.tag)

    def M(self):
        """Mass matrix."""
        mx = self.mx
        my = self.my
        # fmt: off
        M = np.array([[mx, 0],
                      [0, my]])
        # fmt: on

        return M

    def C(self):
        """Damping coefficients matrix."""
        return np.zeros((2, 2))

    def K(self):
        """Stiffness coefficients matrix."""
        return np.zeros((2, 2))

    def G(self):
        """Gyroscopic matrix."""
        return np.zeros((2, 2))

    def dof_mapping(self):
        return dict(x_0=0, y_0=1)
