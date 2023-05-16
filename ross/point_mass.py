"""Point mass module.

This module defines the PointMass class which will be used to link elements.
"""
import os
from pathlib import Path

import numpy as np
import toml
from plotly import graph_objects as go

from ross.element import Element
from ross.units import check_units

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
    m: float, pint.Quantity, optional
        Mass for the element.
    mx: float, pint.Quantity, optional
        Mass for the element on the x direction.
    my: float, pint.Quantity, optional
        Mass for the element on the y direction.
    tag: str
        A tag to name the element
    color : str, optional
        A color to be used when the element is represented.
        Default is "DarkSalmon".

    Examples
    --------
    >>> p0 = PointMass(n=0, m=2)
    >>> p0.M()
    array([[2., 0.],
           [0., 2.]])
    >>> p1 = PointMass(n=0, mx=2, my=3)
    >>> p1.M()
    array([[2., 0.],
           [0., 3.]])
    """

    @check_units
    def __init__(self, n=None, m=None, mx=None, my=None, tag=None, color="DarkSalmon"):
        self.n = n
        self.m = m

        if mx is None and my is None:
            mx = float(m)
            my = float(m)

        self.mx = float(mx)
        self.my = float(my)
        self.tag = tag
        self.dof_global_index = None
        self.color = color

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other : obj
            parameter for comparasion

        Returns
        -------
        True if other is equal to the reference parameter.
        False if not.

        Example
        -------
        >>> pointmass1 = point_mass_example()
        >>> pointmass2 = point_mass_example()
        >>> pointmass1 == pointmass2
        True
        """
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __repr__(self):
        """Return a string representation of a point mass element.

        Returns
        -------
        A string representation of a point mass element object.

        Examples
        --------
        >>> point_mass = point_mass_example()
        >>> point_mass
        PointMass(n=0, mx=1.0, my=2.0, tag='pointmass')
        """
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n}, mx={self.mx:{0}.{5}},"
            f" my={self.my:{0}.{5}}, tag={self.tag!r})"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings

        Example
        -------
        >>> print(PointMass(n=0, mx=2.5, my=3.25, tag="PointMass"))
        Tag:              PointMass
        Node:             0
        Mass X dir. (kg): 2.5
        Mass Y dir. (kg): 3.25
        """
        return (
            f"Tag:              {self.tag}"
            f"\nNode:             {self.n}"
            f"\nMass X dir. (kg): {self.mx:{2}.{5}}"
            f"\nMass Y dir. (kg): {self.my:{2}.{5}}"
        )

    def M(self):
        """Mass matrix for an instance of a point mass element.

        This method will return the mass matrix for an instance of a point mass element.

        Returns
        -------
        M : np.ndarray
            A matrix of floats containing the values of the mass matrix.

        Examples
        --------
        >>> p1 = PointMass(n=0, mx=2, my=3)
        >>> p1.M()
        array([[2., 0.],
               [0., 3.]])
        """
        mx = self.mx
        my = self.my
        # fmt: off
        M = np.array([[mx, 0],
                      [0, my]])
        # fmt: on

        return M

    def C(self):
        """Damping matrix for an instance of a point mass element.

        This method will return the damping matrix for an instance of a point mass
        element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> p1 = PointMass(n=0, mx=2, my=3)
        >>> p1.C()
        array([[0., 0.],
               [0., 0.]])
        """
        C = np.zeros((2, 2))
        return C

    def K(self):
        """Stiffness matrix for an instance of a point mass element.

        This method will return the stiffness matrix for an instance of a point mass
        element.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> p1 = PointMass(n=0, mx=2, my=3)
        >>> p1.K()
        array([[0., 0.],
               [0., 0.]])
        """
        K = np.zeros((2, 2))
        return K

    def G(self):
        """Gyroscopic matrix for an instance of a point mass element.

        This method will return the gyroscopic matrix for an instance of a point mass
        element.

        Returns
        -------
        G : np.ndarray
            A matrix of floats containing the values of the gyroscopic matrix.

        Examples
        --------
        >>> p1 = PointMass(n=0, mx=2, my=3)
        >>> p1.G()
        array([[0., 0.],
               [0., 0.]])
        """
        G = np.zeros((2, 2))
        return G

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        The numbering of the degrees of freedom for each node.

        Being the following their ordering for a node:

        x_0 - horizontal translation
        y_0 - vertical translation

        >>> p1 = PointMass(n=0, mx=2, my=3)
        >>> p1.dof_mapping()
        {'x_0': 0, 'y_0': 1}
        """
        return dict(x_0=0, y_0=1)

    def _patch(self, position, fig):
        """Point mass element patch.

        Patch that will be used to draw the point mass element using Plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        zpos, ypos = position
        radius = ypos / 12

        customdata = [self.n, self.mx, self.my]
        hovertemplate = (
            f"PointMass Node: {customdata[0]}<br>"
            + f"Mass (X): {customdata[1]:.3f}<br>"
            + f"Mass (Y): {customdata[2]:.3f}<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=[zpos, zpos],
                y=[ypos, -ypos],
                customdata=[customdata] * 2,
                text=hovertemplate,
                mode="markers",
                marker=dict(size=5.0, color=self.color),
                showlegend=False,
                name=self.tag,
                legendgroup="pointmass",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=ypos - radius,
                x1=zpos + radius,
                y1=ypos + radius,
                fillcolor=self.color,
                line_color="black",
            )
        )
        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=-ypos - radius,
                x1=zpos + radius,
                y1=-ypos + radius,
                fillcolor=self.color,
                line_color="black",
            )
        )

        return fig


def point_mass_example():
    """Create an example of point mass element.

    This function returns an instance of a simple point mass. The purpose is to make
    available a simple model so that doctest can be written using it.

    Returns
    -------
    point_mass : ross.PointMass
        An instance of a point mass object.

    Examples
    --------
    >>> pointmass = point_mass_example()
    >>> pointmass.mx
    1.0
    """
    n = 0
    mx = 1.0
    my = 2.0
    point_mass = PointMass(n=n, mx=mx, my=my, tag="pointmass")
    return point_mass
