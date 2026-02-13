"""Coupling Element module.

This module defines the CouplingElement class which will be used to represent the coupling
between two rotor shafts, which add mainly stiffness, mass and inertia to the system.
"""

import inspect

import numpy as np
from plotly import graph_objects as go

from ross.shaft_element import ShaftElement
from ross.units import Q_, check_units

__all__ = ["CouplingElement"]


class CouplingElement(ShaftElement):
    """A coupling element.

    This class creates a coupling element from input data of inertia and mass
    from the left station and right station, and also translational and rotational
    stiffness and damping values. The matrices will be defined considering the
    same local coordinate vector of the `ShaftElement`.

    Parameters
    ----------
    m_l : float, pint.Quantity
        Mass of the left station of coupling element (kg).
    m_r : float, pint.Quantity
        Mass of the right station of coupling element (kg).
    Ip_l : float, pint.Quantity
        Polar moment of inertia of the left station of the coupling element (kg).
    Ip_r : float, pint.Quantity
        Polar moment of inertia of the right station of the coupling element (kg.m²).
    Id_l : float, pint.Quantity, optional
        Diametral moment of inertia of the left station of the coupling element (kg.m²).
        If not given, it is assumed to be half of `Ip_l`.
    Id_r : float, pint.Quantity, optional
        Diametral moment of inertia of the right station of the coupling element (kg.m²).
        If not given, it is assumed to be half of `Ip_r`.
    kt_x : float, optional
        Translational stiffness in `x` (N/m).
        Default is 0.
    kt_y : float, optional
        Translational stiffness in `y` (N/m).
        Default is 0.
    kt_z : float, optional
        Axial stiffness (N/m).
        Default is 0.
    kr_x : float, optional
        Rotational stiffness in `x` (N.m/rad).
        Default is 0.
    kr_y : float, optional
        Rotational stiffness in `y` (N.m/rad).
        Default is 0.
    kr_z : float, optional
        Torsional stiffness (N.m/rad).
        Default is 0.
    ct_x : float, optional
        Translational damping in `x` (N.s/m).
        Default is 0.
    ct_y : float, optional
        Translational damping in `y` (N.s/m).
        Default is 0.
    ct_z : float, optional
        Axial damping (N.s/m).
        Default is 0.
    cr_x : float, optional
        Rotational damping in `x` (N.m.s/rad).
        Default is 0.
    cr_y : float, optional
        Rotational damping in `y` (N.m.s/rad).
        Default is 0.
    cr_z : float, optional
        Torsional damping (N.m.s/rad).
        Default is 0.
    o_d : float, optional
        Outer diameter (m). This parameter is primarily used for visualization
        purposes and does not affect calculations.
    L : float, optional
        Element length (m). This parameter is primarily used for visualization
        purposes and does not affect calculations.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float or str, optional
        The scale factor is used to scale the coupling drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#647e91'.

    Examples
    --------
    >>> # Coupling element with torsional stiffness
    >>> m = 151.55
    >>> Ip = 2.197
    >>> torsional_stiffness = 3.04256e6
    >>> coupling = CouplingElement(
    ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
    ...            kr_z=torsional_stiffness
    ... )
    >>> coupling.Id_l
    0.54925
    """

    @check_units
    def __init__(
        self,
        m_l,
        m_r,
        Ip_l,
        Ip_r,
        Id_l=0,
        Id_r=0,
        kt_x=0,
        kt_y=0,
        kt_z=0,
        kr_x=0,
        kr_y=0,
        kr_z=0,
        ct_x=0,
        ct_y=0,
        ct_z=0,
        cr_x=0,
        cr_y=0,
        cr_z=0,
        o_d=None,
        L=None,
        n=None,
        tag=None,
        scale_factor=1,
        color="#647e91",
    ):
        self.n = n
        self.n_l = n
        self.n_r = None
        if n is not None:
            self.n_r = n + 1

        self.m_l = float(m_l)
        self.m_r = float(m_r)
        self.m = self.m_l + self.m_r

        self.Ip_l = float(Ip_l)
        self.Ip_r = float(Ip_r)
        self.Ip = self.Ip_l + self.Ip_r
        self.Id_l = float(Id_l) if Id_l else Ip_l / 2
        self.Id_r = float(Id_r) if Id_r else Ip_r / 2

        self.kt_x = float(kt_x)
        self.kt_y = float(kt_y)
        self.kt_z = float(kt_z)

        self.kr_x = float(kr_x)
        self.kr_y = float(kr_y)
        self.kr_z = float(kr_z)

        self.ct_x = float(ct_x)
        self.ct_y = float(ct_y)
        self.ct_z = float(ct_z)

        self.cr_x = float(cr_x)
        self.cr_y = float(cr_y)
        self.cr_z = float(cr_z)

        self.o_d = 0.2 if o_d is None else float(o_d)
        self.L = 0.2 if L is None else float(L)
        self.odl = self.o_d
        self.odr = self.o_d

        self.tag = tag
        self.scale_factor = scale_factor
        self.color = color
        self.dof_global_index = None

        self.beam_cg = self.L / 2
        self.Im = 1 / 8 * self.m * self.o_d**2
        self.slenderness_ratio = self.L / self.o_d

    def __repr__(self):
        """Return a string representation of a coupling element.

        Returns
        -------
        A string representation of a coupling element object.

        Examples
        --------
        >>> m = 151.55
        >>> Ip = 2.197
        >>> torsional_stiffness = 3.04256e6
        >>> coupling = CouplingElement(
        ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
        ...            kr_z=torsional_stiffness
        ... )
        >>> coupling # doctest: +ELLIPSIS
        CouplingElement(m=151.55, Ip=2.197...
        """
        return (
            f"{self.__class__.__name__}"
            f"(m={self.m:{0}.{5}}, "
            f"Ip={self.Ip:{0}.{5}}, "
            f"n={self.n})"
        )

    def save(self, file):
        from ross.utils import load_data, dump_data

        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}

        try:
            data = load_data(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}_{self.tag}"] = args
        dump_data(data, file)

    @classmethod
    def read_toml_data(cls, data):
        return cls(**data)

    def M(self):
        """Mass matrix for an instance of a coupling element.

        This method will return the mass matrix for an instance of a coupling element.

        Returns
        -------
        M : np.ndarray
            A matrix of floats containing the values of the mass matrix.

        Examples
        --------
        >>> m = 151.55
        >>> Ip = 2.197
        >>> torsional_stiffness = 3.04256e6
        >>> coupling = CouplingElement(
        ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
        ...            kr_z=torsional_stiffness
        ... )
        >>> coupling.M()[:6, :6]
        array([[75.775  ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
               [ 0.     , 75.775  ,  0.     ,  0.     ,  0.     ,  0.     ],
               [ 0.     ,  0.     , 75.775  ,  0.     ,  0.     ,  0.     ],
               [ 0.     ,  0.     ,  0.     ,  0.54925,  0.     ,  0.     ],
               [ 0.     ,  0.     ,  0.     ,  0.     ,  0.54925,  0.     ],
               [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.0985 ]])
        """
        m = self.m_l
        Id = self.Id_l
        Ip = self.Ip_l
        # fmt: off
        Ml = np.array([
            [m,  0,  0,  0,  0,  0],
            [0,  m,  0,  0,  0,  0],
            [0,  0,  m,  0,  0,  0],
            [0,  0,  0, Id,  0,  0],
            [0,  0,  0,  0, Id,  0],
            [0,  0,  0,  0,  0, Ip],
        ])
        # fmt: on

        m = self.m_r
        Id = self.Id_r
        Ip = self.Ip_r
        # fmt: off
        Mr = np.array([
            [m,  0,  0,  0,  0,  0],
            [0,  m,  0,  0,  0,  0],
            [0,  0,  m,  0,  0,  0],
            [0,  0,  0, Id,  0,  0],
            [0,  0,  0,  0, Id,  0],
            [0,  0,  0,  0,  0, Ip],
        ])
        # fmt: on

        M = np.zeros((12, 12))
        M[:6, :6] = Ml
        M[6:, 6:] = Mr

        return M

    def K(self):
        """Stiffness matrix for an instance of a coupling element.

        This method will return the stiffness matrix for an instance of a coupling
        element.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> m = 151.55
        >>> Ip = 2.197
        >>> torsional_stiffness = 3.04256e6
        >>> coupling = CouplingElement(
        ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
        ...            kr_z=torsional_stiffness
        ... )
        >>> coupling.K()[:6, :6]
        array([[      0.,       0.,       0.,       0.,       0.,       0.],
               [      0.,       0.,       0.,       0.,       0.,       0.],
               [      0.,       0.,       0.,       0.,       0.,       0.],
               [      0.,       0.,       0.,       0.,       0.,       0.],
               [      0.,       0.,       0.,       0.,       0.,       0.],
               [      0.,       0.,       0.,       0.,       0., 3042560.]])
        """
        k1 = self.kt_x
        k2 = self.kt_y
        k3 = self.kt_z
        k4 = self.kr_x
        k5 = self.kr_y
        k6 = self.kr_z
        # fmt: off
        K = np.array([
            [  k1,    0,    0,    0,    0,    0, -k1,    0,   0,    0,   0,    0],
            [   0,   k2,    0,    0,    0,    0,   0,  -k2,   0,    0,   0,    0],
            [   0,    0,   k3,    0,    0,    0,   0,    0, -k3,    0,   0,    0],
            [   0,    0,    0,   k4,    0,    0,   0,    0,   0,  -k4,   0,    0],
            [   0,    0,    0,    0,   k5,    0,   0,    0,   0,    0, -k5,    0],
            [   0,    0,    0,    0,    0,   k6,   0,    0,   0,    0,   0,  -k6],
            [ -k1,    0,    0,    0,    0,    0,  k1,    0,   0,    0,   0,    0],
            [   0,  -k2,    0,    0,    0,    0,   0,   k2,   0,    0,   0,    0],
            [   0,    0,  -k3,    0,    0,    0,   0,    0,  k3,    0,   0,    0],
            [   0,    0,    0,  -k4,    0,    0,   0,    0,   0,   k4,   0,    0],
            [   0,    0,    0,    0,  -k5,    0,   0,    0,   0,    0,  k5,    0],
            [   0,    0,    0,    0,    0,  -k6,   0,    0,   0,    0,   0,   k6],
        ])
        # fmt: on

        return K

    def Kst(self):
        return np.zeros((12, 12))

    def C(self):
        """Damping matrix for an instance of a coupling element.

        This method will return the damping matrix for an instance of a coupling
        element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> m = 151.55
        >>> Ip = 2.197
        >>> torsional_stiffness = 3.04256e6
        >>> coupling = CouplingElement(
        ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
        ...            kr_z=torsional_stiffness
        ... )
        >>> coupling.C()[:6, :6]
        array([[0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
        """
        c1 = self.ct_x
        c2 = self.ct_y
        c3 = self.ct_z
        c4 = self.cr_x
        c5 = self.cr_y
        c6 = self.cr_z
        # fmt: off
        C = np.array([
            [  c1,    0,    0,    0,    0,    0, -c1,    0,   0,    0,   0,    0],
            [   0,   c2,    0,    0,    0,    0,   0,  -c2,   0,    0,   0,    0],
            [   0,    0,   c3,    0,    0,    0,   0,    0, -c3,    0,   0,    0],
            [   0,    0,    0,   c4,    0,    0,   0,    0,   0,  -c4,   0,    0],
            [   0,    0,    0,    0,   c5,    0,   0,    0,   0,    0, -c5,    0],
            [   0,    0,    0,    0,    0,   c6,   0,    0,   0,    0,   0,  -c6],
            [ -c1,    0,    0,    0,    0,    0,  c1,    0,   0,    0,   0,    0],
            [   0,  -c2,    0,    0,    0,    0,   0,   c2,   0,    0,   0,    0],
            [   0,    0,  -c3,    0,    0,    0,   0,    0,  c3,    0,   0,    0],
            [   0,    0,    0,  -c4,    0,    0,   0,    0,   0,   c4,   0,    0],
            [   0,    0,    0,    0,  -c5,    0,   0,    0,   0,    0,  c5,    0],
            [   0,    0,    0,    0,    0,  -c6,   0,    0,   0,    0,   0,   c6],
        ])
        # fmt: on

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a coupling element.

        This method will return the gyroscopic matrix for an instance of a coupling
        element.

        Returns
        -------
        G: np.ndarray
            Gyroscopic matrix for the coupling element.

        Examples
        --------
        >>> m = 151.55
        >>> Ip = 2.197
        >>> torsional_stiffness = 3.04256e6
        >>> coupling = CouplingElement(
        ...            m_l=m / 2, m_r=m / 2, Ip_l=Ip / 2, Ip_r=Ip / 2,
        ...            kr_z=torsional_stiffness
        ... )
        >>> coupling.G()[:6, :6]
        array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  0.    ,  1.0985,  0.    ],
               [ 0.    ,  0.    ,  0.    , -1.0985,  0.    ,  0.    ],
               [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]])
        """
        Ip = self.Ip_l
        # fmt: off
        Gl = np.array([
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0, Ip, 0],
            [0, 0, 0, -Ip,  0, 0],
            [0, 0, 0,   0,  0, 0],
        ])
        # fmt: on

        Ip = self.Ip_r
        # fmt: off
        Gr = np.array([
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0, Ip, 0],
            [0, 0, 0, -Ip,  0, 0],
            [0, 0, 0,   0,  0, 0],
        ])
        # fmt: on

        G = np.zeros((12, 12))
        G[:6, :6] = Gl
        G[6:, 6:] = Gr

        return G

    def _patch(self, position, check_sld, fig, units):
        """Coupling element patch.

        Patch that will be used to draw the coupling element using Plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            This parameter makes no difference in the coupling element,
            only in the shaft element.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        units : str, optional
            Element length units.
            Default is 'm'.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        zpos, ypos = position
        L = self.L
        od = self.o_d * self.scale_factor

        # plot the coupling
        z_upper = [zpos, zpos, zpos + L, zpos + L, zpos]
        y_upper = [0, od / 2, od / 2, 0, 0]
        z_lower = [zpos, zpos, zpos + L, zpos + L, zpos]
        y_lower = [0, -od / 2, -od / 2, 0, 0]

        z_pos = z_upper
        z_pos.extend(z_lower)

        y_pos = y_upper
        y_pos.extend(y_lower)

        name = (
            f"{self.tag}<br>(<i>CouplingElement</i>)"
            if "ShaftElement" in self.tag
            else self.tag
        )

        legend = "Coupling"

        customdata = [
            self.n,
            self.m,
            self.Ip,
        ]

        hovertemplate = (
            f"Element Number: {customdata[0]}<br>"
            + f"Mass: {customdata[1]:.3f} kg<br>"
            + f"Polar moment of inertia: {customdata[2]:.3f} kg⋅m²<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=Q_(z_pos, "m").to(units).m,
                y=Q_(y_pos, "m").to(units).m,
                customdata=[customdata] * len(z_pos),
                text=hovertemplate,
                mode="lines",
                opacity=0.5,
                fill="toself",
                fillcolor=self.color,
                line=dict(width=1.5, color="black", dash="dash"),
                showlegend=False,
                name=name,
                legendgroup=legend,
                hoveron="points+fills",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        return fig
