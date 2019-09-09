"""Point mass module.

This module defines the PointMass class which will be used to link elements.
"""
import numpy as np
from .element import Element

__all__ = ["PointMass"]


class PointMass(Element):
    """A point mass element.

    This class will create a point mass element.
    This element can be used to link other elements in the analysis.

    Parameters
    ----------
    n: int
        Node which the bearing will be located in.
    m: float
        Mass for the element.
    """

    def __init__(self, n=None, m=None):

        self.n = n
        self.m = m

    def M(self):
        """Mass matrix."""
        m = self.m
        # fmt: off
        M = np.array([[m, 0],
                      [0, m]])
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
        return dict(x0=0, y0=1)
