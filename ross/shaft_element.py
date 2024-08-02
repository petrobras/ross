"""Shaft Element module.

This module defines the ShaftElement classes which will be used to represent the rotor
shaft. There're 2 options, an element with 8 or 12 degrees of freedom.
"""

import inspect
import os
from pathlib import Path

import numpy as np
import toml
from plotly import graph_objects as go

from ross.element import Element
from ross.materials import Material, steel
from ross.units import Q_, check_units
from ross.utils import read_table_file

__all__ = ["ShaftElement", "ShaftElement6DoF"]


class ShaftElement(Element):
    r"""A shaft element.

    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects. The object can be
    cylindrical or conical and the formulation is based on :cite:`genta1988conical`
    The matrices will be defined considering the following local
    coordinate vector:

    .. math::

        [x_0, y_0, \alpha_0, \beta_0, x_1, y_1, \alpha_1, \beta_1]^T
    Where :math:`\alpha_0` and :math:`\alpha_1` are the bending on the yz plane
    and :math:`\beta_0` and :math:`\beta_1` are the bending on the xz plane.

    Parameters
    ----------
    L : float, pint.Quantity
        Element length (m).
    idl : float, pint.Quantity
        Inner diameter of the element at the left position (m).
    odl : float, pint.Quantity
        Outer diameter of the element at the left position (m).
    idr : float, pint.Quantity, optional
        Inner diameter of the element at the right position (m).
        Default is equal to idl value (cylindrical element).
    odr : float, pint.Quantity, optional
        Outer diameter of the element at the right position (m).
        Default is equal to odl value (cylindrical element).
    material : ross.Material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force (N).
    torque : float, optional
        Torque (N*m).
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : str, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'
    alpha : float, optional
        Mass proportional damping factor.
        Default is zero.
    beta : float, optional
        Stiffness proportional damping factor.
        Default is zero.
    tag : str, optional
        Element tag.
        Default is None.

    Returns
    -------
    shaft_element : ross.ShaftElement
        A shaft_element object.

    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area at half length (m**2).
    A_l : float
        Element section area at left end (m**2).
    A_r : float
        Element section area at right end (m**2).
    beam_cg : float
        Element center of gravity local position (m).
    axial_cg_pos : float
        Element center of gravity global position (m).
        This should be used only after the rotor is built.
        Default is None.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane (m**4).
    phi : float
        Constant that is used according to :cite:`friswell2010dynamics` to
        consider rotary inertia and shear effects. If these are not considered
        :math:`\phi=0`.
    kappa : float
        Shear coefficient for the element.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> from ross.materials import steel
    >>> # Euler-Bernoulli conical element
    >>> Euler_Bernoulli_Element = ShaftElement(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         idr=0.05, odr=0.15,
    ...                         rotary_inertia=False,
    ...                         shear_effects=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> # Timoshenko cylindrical element. In this case idr and odr are omitted.
    >>> Timoshenko_Element = ShaftElement(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    @check_units
    def __init__(
        self,
        L,
        idl,
        odl,
        idr=None,
        odr=None,
        material=None,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        shear_method_calc="cowper",
        tag=None,
        alpha=0,
        beta=0,
    ):
        if idr is None:
            idr = idl
        if odr is None:
            odr = odl

        if material is None:
            raise AttributeError("Material is not defined.")

        if type(material) is str:
            self.material = Material.load_material(material)
        else:
            self.material = material

        self.shear_effects = shear_effects
        self.rotary_inertia = rotary_inertia
        self.gyroscopic = gyroscopic
        self.axial_force = axial_force
        self.torque = torque
        self._n = n
        self.n_l = n
        self.n_r = None
        if n is not None:
            self.n_r = n + 1

        self.tag = tag
        self.shear_method_calc = shear_method_calc

        self.L = float(L)
        self.o_d = (float(odl) + float(odr)) / 2
        self.i_d = (float(idl) + float(idr)) / 2
        self.idl = float(idl)
        self.odl = float(odl)
        self.idr = float(idr)
        self.odr = float(odr)
        self.color = self.material.color

        self.alpha = float(alpha)
        self.beta = float(beta)

        # A_l = cross section area from the left side of the element
        # A_r = cross section area from the right side of the element
        A_l = np.pi * (odl**2 - idl**2) / 4
        A_r = np.pi * (odr**2 - idr**2) / 4
        self.A_l = A_l
        self.A_r = A_r

        # Second moment of area of the cross section from the left side
        # of the element
        Ie_l = np.pi * (odl**4 - idl**4) / 64
        Ie_r = np.pi * (odr**4 - idr**4) / 64
        self.Ie_l = Ie_l
        self.Ie_r = Ie_r

        outer = self.odl**2 + self.odl * self.odr + self.odr**2
        inner = self.idl**2 + self.idl * self.idr + self.idr**2
        self.volume = np.pi * (self.L / 12) * (outer - inner)
        self.m = self.material.rho * self.volume

        roj = odl / 2
        rij = idl / 2
        rok = odr / 2
        rik = idr / 2

        # geometrical coefficients
        delta_ro = rok - roj
        delta_ri = rik - rij
        a1 = 2 * np.pi * (roj * delta_ro - rij * delta_ri) / A_l
        a2 = np.pi * (roj**3 * delta_ro - rij**3 * delta_ri) / Ie_l
        b1 = np.pi * (delta_ro**2 - delta_ri**2) / A_l
        b2 = 3 * np.pi * (roj**2 * delta_ro**2 - rij**2 * delta_ri**2) / (2 * Ie_l)
        gama = np.pi * (roj * delta_ro**3 - rij * delta_ri**3) / Ie_l
        delta = np.pi * (delta_ro**4 - delta_ri**4) / (4 * Ie_l)

        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.gama = gama
        self.delta = delta

        # the area is calculated from the cross section located in the middle
        # of the element
        self.A = A_l * (1 + a1 * 0.5 + b1 * 0.5**2)

        # Ie is the second moment of area of the cross section - located in
        # the middle of the element - about the neutral plane
        Ie = Ie_l * (1 + a2 * 0.5 + b2 * 0.5**2 + gama * 0.5**3 + delta * 0.5**4)
        self.Ie = Ie

        phi = 0

        # geometric center
        c1 = roj**2 + 2 * roj * rok + 3 * rok**2 - rij**2 - 2 * rij * rik - 3 * rik**2
        c2 = (roj**2 + roj * rok + rok**2) - (rij**2 + rij * rik + rik**2)
        self.beam_cg = L * c1 / (4 * c2)
        self.axial_cg_pos = None

        # Slenderness ratio of beam elements (G*A*L**2) / (E*I)
        sld = (self.material.G_s * self.A * self.L**2) / (self.material.E * Ie)
        self.slenderness_ratio = sld

        # Moment of inertia
        # fmt: off
        self.Im = (
            (np.pi * L * (self.m / self.volume) / 10) *
            ((roj ** 4 + roj ** 3 * rok + roj ** 2 * rok ** 2 + roj * rok ** 3 + rok ** 4) -
             (rij ** 4 + rij ** 3 * rik + rij ** 2 * rik ** 2 + rij * rik ** 3 + rik ** 4))
        )
        # fmt: on

        # picking a method to calculate the shear coefficient
        # List of avaible methods:
        # hutchinson - kappa as per Hutchinson (2001)
        # cowper - kappa as per Cowper (1996)
        if shear_effects:
            r = ((idl + idr) / 2) / ((odl + odr) / 2)
            r2 = r * r
            r12 = (1 + r2) ** 2
            if shear_method_calc == "hutchinson":
                # Shear coefficient (phi)
                # kappa as per Hutchinson (2001)
                # fmt: off
                kappa = 6 * r12 * ((1 + self.material.Poisson) /
                        ((r12 * (7 + 12 * self.material.Poisson + 4 * self.material.Poisson ** 2) +
                        4 * r2 * (5 + 6 * self.material.Poisson + 2 * self.material.Poisson ** 2))))
                # fmt: on
            elif shear_method_calc == "cowper":
                # kappa as per Cowper (1996)
                # fmt: off
                kappa = 6 * r12 * (
                    (1 + self.material.Poisson)
                    / (r12 * (7 + 6 * self.material.Poisson) + r2 * (20 + 12 * self.material.Poisson))
                )
                # fmt: on
            else:
                raise Warning(
                    "This method of calculating shear coefficients is not implemented. See guide for futher informations."
                )

            # fmt: off
            phi = 12 * self.material.E * self.Ie / (self.material.G_s * kappa * self.A * L ** 2)
            # fmt: on
            self.kappa = kappa

        self.phi = phi
        self.dof_global_index = None

    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other : obj
            parameter for comparasion

        Returns
        -------
        bool
            True if the comparison is true; False otherwise.

        Example
        -------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(
        ...        L=0.25, idl=0, idr=0, odl=0.05, odr=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft2 = ShaftElement(
        ...        L=0.25, idl=0, idr=0, odl=0.05, odr=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 == shaft2
        True
        """
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __repr__(self):
        """Return a string representation of a shaft element.

        Returns
        -------
        A string representation of a shaft element object.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(
        ...        L=0.25, idl=0, idr=0, odl=0.05, odr=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 # doctest: +ELLIPSIS
        ShaftElement(L=0.25, idl=0.0...
        """
        return (
            f"{self.__class__.__name__}"
            f"(L={self.L:{0}.{5}}, idl={self.idl:{0}.{5}}, "
            f"idr={self.idr:{0}.{5}}, odl={self.odl:{0}.{5}},  "
            f"odr={self.odr:{0}.{5}}, material={self.material.name!r}, "
            f"n={self.n})"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings

        Example
        -------
        >>> print(ShaftElement(L=0.25, idl=0, odl=0.05, odr=0.08, material=steel))
        Element Number:             None
        Element Lenght   (m):       0.25
        Left Int. Diam.  (m):        0.0
        Left Out. Diam.  (m):       0.05
        Right Int. Diam. (m):        0.0
        Right Out. Diam. (m):       0.08
        -----------------------------------
        Steel
        -----------------------------------
        Density         (kg/m**3): 7810.0
        Young`s modulus (N/m**2):  2.11e+11
        Shear modulus   (N/m**2):  8.12e+10
        Poisson coefficient     :  0.29926108
        """
        return (
            f"Element Number:             {self.n}"
            f"\nElement Lenght   (m): {self.L:{10}.{5}}"
            f"\nLeft Int. Diam.  (m): {self.idl:{10}.{5}}"
            f"\nLeft Out. Diam.  (m): {self.odl:{10}.{5}}"
            f"\nRight Int. Diam. (m): {self.idr:{10}.{5}}"
            f"\nRight Out. Diam. (m): {self.odr:{10}.{5}}"
            f'\n{35*"-"}'
            f"\n{self.material}"
        )

    def __hash__(self):
        return hash(self.tag)

    def save(self, file):
        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}

        # add material characteristics so that the shaft element can be reconstructed
        # even if the material is not in the available_materials file.
        args["material"] = {
            "name": self.material.name,
            "rho": self.material.rho,
            "E": self.material.E,
            "G_s": self.material.G_s,
            "color": self.material.color,
        }

        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}_{self.tag}"] = args
        with open(file, "w") as f:
            toml.dump(data, f)

    @classmethod
    def read_toml_data(cls, data):
        data["material"] = Material(**data["material"])
        return cls(**data)

    @property
    def n(self):
        """Set the element number as property.

        Returns
        -------
        n : int
            Element number
        """
        return self._n

    @n.setter
    def n(self, value):
        """Set a new value for the element number.

        Parameters
        ----------
        value : int
            element number

        Example
        -------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(
        ...        L=0.25, idl=0, idr=0, odl=0.05, odr=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.n = 0
        >>> shaft1 # doctest: +ELLIPSIS
        ShaftElement(L=0.25, idl=0.0...
        """
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

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
        z_0 - axial translation
        alpha_0 - rotation around horizontal
        beta_0  - rotation around vertical
        theta_0 - torsion around axial

        >>> sh = ShaftElement(L=0.5, idl=0.05, odl=0.1, material=steel,
        ...                   rotary_inertia=True, shear_effects=True)
        >>> sh.dof_mapping()["x_0"]
        0
        """
        return dict(
            x_0=0, y_0=1, alpha_0=2, beta_0=3, x_1=4, y_1=5, alpha_1=6, beta_1=7
        )

    def M(self):
        """Mass matrix for an instance of a shaft element.

        Returns
        -------
        M : np.ndarray
            Mass matrix for the shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(
        ...                         L=0.5, idl=0.05, idr=0.05, odl=0.1,
        ...                         odr=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.M()[:4, :4]
        array([[11.36986417,  0.        ,  0.        ,  0.86197637],
               [ 0.        , 11.36986417, -0.86197637,  0.        ],
               [ 0.        , -0.86197637,  0.08667495,  0.        ],
               [ 0.86197637,  0.        ,  0.        ,  0.08667495]])
        """
        phi = self.phi
        L = self.L
        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        delta = self.delta
        gama = self.gama
        Ie_l = self.Ie_l
        A_l = self.A_l

        m1 = (
            (468 + 882 * phi + 420 * phi**2)
            + a1 * (108 + 210 * phi + 105 * phi**2)
            + b1 * (38 + 78 * phi + 42 * phi**2)
        )
        m2 = (
            (66 + 115.5 * phi + 52.5 * phi**2)
            + a1 * (21 + 40.5 * phi + 21 * phi**2)
            + b1 * (8.5 + 18 * phi + 10.5 * phi**2)
        )
        m3 = (
            (162 + 378 * phi + 210 * phi**2)
            + a1 * (81 + 189 * phi + 105 * phi**2)
            + b1 * (46 + 111 * phi + 63 * phi**2)
        )
        m4 = (
            (39 + 94.5 * phi + 52.5 * phi**2)
            + a1 * (18 + 40.5 * phi + 21 * phi**2)
            + b1 * (9.5 + 21 * phi + 10.5 * phi**2)
        )
        m5 = (
            (12 + 21 * phi + 10.5 * phi**2)
            + a1 * (4.5 + 9 * phi + 5.25 * phi**2)
            + b1 * (2 + 4.5 * phi + 3 * phi**2)
        )
        m6 = (
            (39 + 94.5 * phi + 52.5 * phi**2)
            + a1 * (21 + 54 * phi + 31.5 * phi**2)
            + b1 * (12.5 + 34.5 * phi + 21 * phi**2)
        )
        m7 = (
            (9 + 21 * phi + 10.5 * phi**2)
            + a1 * (4.5 + 10.5 * phi + 5.25 * phi**2)
            + b1 * (2.5 + 6 * phi + 3 * phi**2)
        )
        m8 = (
            (468 + 882 * phi + 420 * phi**2)
            + a1 * (360 + 672 * phi + 315 * phi**2)
            + b1 * (290 + 540 * phi + 252 * phi**2)
        )
        m9 = (
            (66 + 115.5 * phi + 52.5 * phi**2)
            + a1 * (45 + 75 * phi + 31.5 * phi**2)
            + b1 * (32.5 + 52.5 * phi + 21 * phi**2)
        )
        m10 = (
            (12 + 21 * phi + 10.5 * phi**2)
            + a1 * (7.5 + 12 * phi + 5.25 * phi**2)
            + b1 * (5 + 7.5 * phi + 3 * phi**2)
        )

        # fmt: off
        Mt = np.array([
                [   m1,     0,        0,    L*m2,     m3,     0,        0,    -L*m4],
                [    0,    m1,    -L*m2,       0,      0,    m3,     L*m4,        0],
                [    0, -L*m2,  L**2*m5,       0,      0, -L*m6, -L**2*m7,        0],
                [ L*m2,     0,        0,  L**2*m5,  L*m6,     0,        0, -L**2*m7],
                [   m3,     0,        0,     L*m6,    m8,     0,        0,    -L*m9],
                [    0,    m3,    -L*m6,        0,     0,    m8,     L*m9,        0],
                [    0,  L*m4, -L**2*m7,        0,     0,  L*m9, L**2*m10,        0],
                [-L*m4,     0,        0, -L**2*m7, -L*m9,     0,        0, L**2*m10],
        ])
        # fmt: on
        M = self.material.rho * A_l * L * Mt / (1260 * (1 + phi) ** 2)

        if self.rotary_inertia:
            # fmt: off
            m11 = 252 + 126 * a2 + 72 * b2 + 45 * gama + 30 * delta
            m12 = (
                21 - 105 * phi
                + a2 * (21 - 42 * phi)
                + b2 * (15 - 21 * phi)
                + gama * (10.5 - 12 * phi)
                + delta * (7.5 - 7.5 * phi)
            )
            m13 = (
                21 - 105 * phi
                - 63 * a2 * phi
                - b2 * (6 + 42 * phi)
                - gama * (7.5 + 30 * phi)
                - delta * (7.5 + 22.5 * phi)
            )
            m14 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (7 - 7 * phi + 17.5 * phi ** 2)
                + b2 * (4 - 7 * phi + 7 * phi ** 2)
                + gama * (2.75 - 5 * phi + 3.5 * phi ** 2)
                + delta * (2 - 3.5 * phi + 2 * phi ** 2)
            )
            m15 = (
                7 + 35 * phi - 35 * phi ** 2
                + a2 * (3.5 + 17.5 * phi - 17.5 * phi ** 2)
                + b2 * (3 + 10.5 * phi - 10.5 * phi ** 2)
                + gama * (2.75 + 7 * phi - 7 * phi ** 2)
                + delta * (2.5 + 5 * phi - 5 * phi ** 2)
            )
            m16 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (21 + 42 * phi + 52.5 * phi ** 2)
                + b2 * (18 + 42 * phi + 42 * phi ** 2)
                + gama * (16.25 + 40 * phi + 35 * phi ** 2)
                + delta * (15 + 37.5 * phi + 30 * phi ** 2)
            )

            Mr = np.array([
                    [  m11,      0,         0,     L*m12,   -m11,     0,         0,     L*m13],
                    [    0,    m11,    -L*m12,         0,      0,  -m11,    -L*m13,         0],
                    [    0, -L*m12,  L**2*m14,         0,      0, L*m12, -L**2*m15,         0],
                    [L*m12,      0,         0,  L**2*m14, -L*m12,     0,         0, -L**2*m15],
                    [ -m11,      0,         0,    -L*m12,    m11,     0,         0,    -L*m13],
                    [    0,   -m11,     L*m12,         0,      0,   m11,     L*m13,         0],
                    [    0, -L*m13, -L**2*m15,         0,      0, L*m13,  L**2*m16,         0],
                    [L*m13,      0,         0, -L**2*m15, -L*m13,     0,         0,  L**2*m16],
            ])
            # fmt: on
            Mr = self.material.rho * Ie_l * Mr / (210 * L * (1 + phi) ** 2)
            M = M + Mr

        return M

    def K(self):
        """Stiffness matrix for an instance of a shaft element.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement(
        ...                         L=0.5, idl=0.05, idr=0.05, odl=0.1,
        ...                         odr=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.K()[:4, :4]/1e6
        array([[219.96144416,   0.        ,   0.        ,  41.29754659],
               [  0.        , 219.96144416, -41.29754659,   0.        ],
               [  0.        , -41.29754659,  12.23526375,   0.        ],
               [ 41.29754659,   0.        ,   0.        ,  12.23526375]])
        """
        L = self.L
        phi = self.phi
        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        delta = self.delta
        gama = self.gama
        A = self.A
        A_l = self.A_l
        Ie_l = self.Ie_l
        E = self.material.E

        # fmt: off
        k1 = 1260 + 630 * a2 + 504 * b2 + 441 * gama + 396 * delta
        k2 = (
            630
            + 210 * a2
            + 147 * b2
            + 126 * gama
            + 114 * delta
            - phi * (105 * a2 + 105 * b2 + 94.5 * gama + 84 * delta)
        )
        k3 = (
            630
            + 420 * a2
            + 357 * b2
            + 315 * gama
            + 282 * delta
            + phi * (105 * a2 + 105 * b2 + 94.5 * gama + 84 * delta)
        )
        k4 = (
            420 + 210 * phi + 105 * phi ** 2
            + a2 * (105 + 52.5 * phi ** 2)
            + b2 * (56 - 35 * phi + 35 * phi ** 2)
            + gama * (42 - 42 * phi + 26.25 * phi ** 2)
            + delta * (36 - 42 * phi + 21 * phi ** 2)
        )
        k5 = (
            210 - 210 * phi - 105 * phi ** 2
            + a2 * (105 - 105 * phi - 52.5 * phi ** 2)
            + b2 * (91 - 70 * phi - 35 * phi ** 2)
            + gama * (84 - 52.5 * phi - 26.25 * phi ** 2)
            + delta * (78 - 42 * phi - 21 * phi ** 2)
        )
        k6 = (
            420 + 210 * phi + 105 * phi ** 2
            + a2 * (315 + 210 * phi + 52.5 * phi ** 2)
            + b2 * (266 + 175 * phi + 35 * phi ** 2)
            + gama * (231 + 147 * phi + 26.25 * phi ** 2)
            + delta * (204 + 126 * phi + 21 * phi ** 2)
        )
        k7 = 12 + 6 * a1 + 4 * b1
        k8 = 6 + 3 * a1 + 2 * b1
        k9 = 3 + 1.5 * a1 + b1

        K1 = np.array([
            [  k1,     0,       0,    L*k2,   -k1,    0,       0,    L*k3],
            [   0,    k1,   -L*k2,       0,     0,  -k1,   -L*k3,       0],
            [   0, -L*k2, L**2*k4,       0,     0, L*k2, L**2*k5,       0],
            [L*k2,     0,       0, L**2*k4, -L*k2,    0,       0, L**2*k5],
            [ -k1,     0,       0,   -L*k2,    k1,    0,       0,   -L*k3],
            [   0,   -k1,    L*k2,       0,     0,   k1,    L*k3,       0],
            [   0, -L*k3, L**2*k5,       0,     0, L*k3, L**2*k6,       0],
            [L*k3,     0,       0, L**2*k5, -L*k3,    0,       0, L**2*k6],
        ])

        K2 = np.array([
            [  k7,     0,       0,    L*k8,   -k7,     0,       0,    L*k8],
            [   0,    k7,   -L*k8,       0,     0,   -k7,   -L*k8,       0],
            [   0, -L*k8, L**2*k9,       0,     0,  L*k8, L**2*k9,       0],
            [L*k8,     0,       0, L**2*k9, -L*k8,     0,       0, L**2*k9],
            [ -k7,     0,       0,   -L*k8,    k7,     0,       0,   -L*k8],
            [   0,   -k7,    L*k8,       0,     0,    k7,    L*k8,       0],
            [   0, -L*k8, L**2*k9,       0,     0,  L*k8, L**2*k9,       0],
            [L*k8,     0,       0, L**2*k9, -L*k8,     0,       0, L**2*k9],
        ])

        K = E * L**(-3) * (1 + phi)**(-2) * (K1 * Ie_l/105 + K2 * self.Ie * phi * A_l / A)

        # axial force
        k10 = 36 + 60 * phi + 30 * phi ** 2
        k11 = L * 3
        k12 = L ** 2 * (4 + 5 * phi + 2.5 * phi ** 2)
        k13 = L ** 2 * (1 + 5 * phi + 2.5 * phi ** 2)

        Kaxial = np.array([
            [ k10,    0,    0,  k11, -k10,    0,    0,  k11],
            [   0,  k10, -k11,    0,    0, -k10, -k11,    0],
            [   0, -k11,  k12,    0,    0,  k11, -k13,    0],
            [ k11,    0,    0,  k12, -k11,    0,    0, -k13],
            [-k10,    0,    0, -k11,  k10,    0,    0, -k11],
            [   0, -k10,  k11,    0,    0,  k10,  k11,    0],
            [   0, -k11, -k13,    0,    0,  k11,  k12,    0],
            [ k11,    0,    0, -k13, -k11,    0,    0,  k12],
        ])

        Kaxial *= self.axial_force / (30 * L * (1 + phi) ** 2)
        K += Kaxial

        # torque
        Ktorque = np.array([
            [   0,    0,    1,    0,    0,    0,   -1,    0],
            [   0,    0,    0,    1,    0,    0,    0,   -1],
            [   1,    0,    0, -L/2,   -1,    0,    0,  L/2],
            [   0,    1,  L/2,    0,    0,   -1, -L/2,    0],
            [   0,    0,   -1,    0,    0,    0,    1,    0],
            [   0,    0,    0,   -1,    0,    0,    0,    1],
            [  -1,    0,    0, -L/2,    1,    0,    0,  L/2],
            [   0,   -1,  L/2,    0,    0,    1, -L/2,    0],
        ])

        Ktorque *= self.torque / L
        K += Ktorque
        # fmt: on

        return K

    def C(self):
        """Stiffness matrix for an instance of a shaft element.

        Returns
        -------
        C : np.array
           Damping matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement(
        ...                         L=0.5, idl=0.05, idr=0.05, odl=0.1,
        ...                         odr=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.C()[:4, :4]
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0., -0.,  0.],
               [ 0., -0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
        """
        # proportional damping matrix
        C = self.alpha * self.M() + self.beta * self.K()

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.

        Returns
        -------
        G : np.ndarray
            Gyroscopic matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(
        ...                         L=0.5, idl=0.05, idr=0.05, odl=0.1,
        ...                         odr=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[ 0.        ,  0.30940809, -0.01085902,  0.        ],
               [-0.30940809,  0.        ,  0.        , -0.01085902],
               [ 0.01085902,  0.        ,  0.        ,  0.0067206 ],
               [ 0.        ,  0.01085902, -0.0067206 ,  0.        ]])
        """
        if self.gyroscopic:
            phi = self.phi
            L = self.L
            a2 = self.a2
            b2 = self.b2
            delta = self.delta
            gama = self.gama
            Ie_l = self.Ie_l

            # fmt: off
            g1 = 252 + 126 * a2 + 72 * b2 + 45 * gama + 30 * delta
            g2 = (
                21 - 105 * phi
                + a2 * (21 - 42 * phi)
                + b2 * (15 - 21 * phi)
                + gama * (10.5 - 12 * phi)
                + delta * (7.5 - 7.5 * phi)
            )
            g3 = (
                21 - 105 * phi
                - 63 * a2 * phi
                - b2 * (6 + 42 * phi)
                - gama * (7.5 + 30 * phi)
                - delta * (7.5 + 22.5 * phi)
            )
            g4 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (7 - 7 * phi + 17.5 * phi ** 2)
                + b2 * (4 - 7 * phi + 7 * phi ** 2)
                + gama * (2.75 - 5 * phi + 3.5 * phi ** 2)
                + delta * (2 - 3.5 * phi + 2 * phi ** 2)
            )
            g5 = (
                7 + 35 * phi - 35 * phi ** 2
                + a2 * (3.5 + 17.5 * phi - 17.5 * phi ** 2)
                + b2 * (3 + 10.5 * phi - 10.5 * phi ** 2)
                + gama * (2.75 + 7 * phi - 7 * phi ** 2)
                + delta * (2.5 + 5 * phi - 5 * phi ** 2)
            )
            g6 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (21 + 42 * phi + 52.5 * phi ** 2)
                + b2 * (18 + 42 * phi + 42 * phi ** 2)
                + gama * (16.25 + 40 * phi + 35 * phi ** 2)
                + delta * (15 + 37.5 * phi + 30 * phi ** 2)
            )

            G = np.array([
                    [   0,    g1,    -L*g2,        0,     0,   -g1,    -L*g3,        0],
                    [ -g1,     0,        0,    -L*g2,    g1,     0,        0,    -L*g3],
                    [L*g2,     0,        0,  L**2*g4, -L*g2,     0,        0, -L**2*g5],
                    [   0,  L*g2, -L**2*g4,        0,     0, -L*g2,  L**2*g5,        0],
                    [   0,   -g1,     L*g2,        0,     0,    g1,     L*g3,        0],
                    [  g1,     0,        0,     L*g2,   -g1,     0,        0,     L*g3],
                    [L*g3,     0,        0, -L**2*g5, -L*g3,     0,        0,  L**2*g6],
                    [   0,  L*g3,  L**2*g5,        0,     0, -L*g3, -L**2*g6,        0],
            ])
            # fmt: on
            G = self.material.rho * Ie_l * 2 * G / (210 * L * (1 + phi) ** 2)

        else:
            G = np.zeros((8, 8))

        return G

    def _patch(self, position, check_sld, fig, units):
        """Shaft element patch.

        Patch that will be used to draw the shaft element using Plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio and color
            the elements in yellow if slenderness ratio < 1.6
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        units : str, optional
            Element length and radius units.
            Default is 'm'.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        if check_sld is True and self.slenderness_ratio < 1.6:
            color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            color = self.material.color
            legend = "Shaft"

        # plot the shaft
        z_upper = [position, position, position + self.L, position + self.L, position]
        y_upper = [self.idl / 2, self.odl / 2, self.odr / 2, self.idr / 2, self.idl / 2]
        z_lower = [position, position, position + self.L, position + self.L, position]
        y_lower = [
            -self.idl / 2,
            -self.odl / 2,
            -self.odr / 2,
            -self.idr / 2,
            -self.idl / 2,
        ]

        z_pos = z_upper
        z_pos.extend(z_lower)

        y_pos = y_upper
        y_pos.extend(y_lower)

        if check_sld:
            customdata = [self.n, self.slenderness_ratio]
            hovertemplate = (
                f"Element Number: {customdata[0]}<br>"
                + f"Slenderness Ratio: {customdata[1]:.3f}"
            )
        else:
            if isinstance(self, ShaftElement6DoF):
                customdata = [
                    self.n,
                    Q_(self.odl, "m").to(units).m,
                    Q_(self.idl, "m").to(units).m,
                    Q_(self.odr, "m").to(units).m,
                    Q_(self.idr, "m").to(units).m,
                    self.alpha,
                    self.beta,
                    Q_(self.L, "m").to(units).m,
                    self.material.name,
                ]
                hovertemplate = (
                    f"Element Number: {customdata[0]}<br>"
                    + f"Left Outer Diameter: {round(customdata[1], 6)} {units}<br>"
                    + f"Left Inner Diameter: {round(customdata[2], 6)} {units}<br>"
                    + f"Right Outer Diameter: {round(customdata[3], 6)} {units}<br>"
                    + f"Right Inner Diameter: {round(customdata[4], 6)} {units}<br>"
                    + f"Alpha Damp. Factor: {round(customdata[5], 6)}<br>"
                    + f"Beta Damp. Factor: {round(customdata[6], 6)}<br>"
                    + f"Element Length: {round(customdata[7], 6)} {units} <br>"
                    + f"Material: {customdata[8]}<br>"
                )
            else:
                customdata = [
                    self.n,
                    Q_(self.odl, "m").to(units).m,
                    Q_(self.idl, "m").to(units).m,
                    Q_(self.odr, "m").to(units).m,
                    Q_(self.idr, "m").to(units).m,
                    Q_(self.L, "m").to(units).m,
                    self.material.name,
                ]
                hovertemplate = (
                    f"Element Number: {customdata[0]}<br>"
                    + f"Left Outer Diameter: {round(customdata[1], 6)} {units}<br>"
                    + f"Left Inner Diameter: {round(customdata[2], 6)} {units}<br>"
                    + f"Right Outer Diameter: {round(customdata[3], 6)} {units}<br>"
                    + f"Right Inner Diameter: {round(customdata[4], 6)} {units}<br>"
                    + f"Element Length: {round(customdata[5], 6)} {units}<br>"
                    + f"Material: {customdata[6]}<br>"
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
                fillcolor=color,
                line=dict(width=1.5, color="black"),
                showlegend=False,
                name=self.tag,
                legendgroup=legend,
                hoveron="points+fills",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=color),
            )
        )

        return fig

    def create_modified(self, **attributes):
        """Return a new shaft element based on the current instance.

        Any attribute passed as an argument will be used to modify the corresponding
        attribute of the instance. Attributes not provided as arguments will retain
        their values from the current instance.

        Parameters
        ----------
        L : float, pint.Quantity, optional
            Element length (m). Default is equal to value of current instance.
        idl : float, pint.Quantity, optional
            Inner diameter of the element at the left position (m).
            Default is equal to value of current instance.
        odl : float, pint.Quantity, optional
            Outer diameter of the element at the left position (m).
            Default is equal to value of current instance.
        idr : float, pint.Quantity, optional
            Inner diameter of the element at the right position (m).
            Default is equal to value of current instance.
        odr : float, pint.Quantity, optional
            Outer diameter of the element at the right position (m).
            Default is equal to value of current instance.
        material : ross.Material, optional
            Shaft material. Default is equal to value of current instance.
        n : int, optional
            Element number (coincident with it's first node).
            Default is equal to value of current instance.
        axial_force : float, optional
            Axial force (N). Default is equal to value of current instance.
        torque : float, optional
            Torque (N*m). Default is equal to value of current instance.
        shear_effects : bool, optional
            Determine if shear effects are taken into account.
            Default is equal to value of current instance.
        rotary_inertia : bool, optional
            Determine if rotary_inertia effects are taken into account.
            Default is equal to value of current instance.
        gyroscopic : bool, optional
            Determine if gyroscopic effects are taken into account.
            Default is equal to value of current instance.
        shear_method_calc : str, optional
            Determines which shear calculation method the user will adopt
            Default is equal to value of current instance.
        alpha : float, optional
            Mass proportional damping factor.
            Default is equal to value of current instance.
        beta : float, optional
            Stiffness proportional damping factor.
            Default is equal to value of current instance.
        tag : str, optional
            Element tag.
            Default is None.

        Returns
        -------
        shaft_element : ross.ShaftElement
            An instance of the modified shaft element.
        """
        return self.__class__(
            L=attributes.get("L", self.L),
            idl=attributes.get("idl", self.idl),
            odl=attributes.get("odl", self.odl),
            idr=attributes.get("idr", self.idr),
            odr=attributes.get("odr", self.odr),
            n=attributes.get("n", self.n),
            material=attributes.get("material", self.material),
            axial_force=attributes.get("axial_force", self.axial_force),
            torque=attributes.get("torque", self.torque),
            rotary_inertia=attributes.get("rotary_inertia", self.rotary_inertia),
            shear_effects=attributes.get("shear_effects", self.shear_effects),
            gyroscopic=attributes.get("gyroscopic", self.gyroscopic),
            shear_method_calc=attributes.get(
                "shear_method_calc", self.shear_method_calc
            ),
            alpha=attributes.get("alpha", self.alpha),
            beta=attributes.get("beta", self.beta),
            tag=attributes.get("tag", None),
        )

    @classmethod
    def from_table(cls, file, sheet_type="Simple", sheet_name=0):
        """Instantiate one or more shafts using inputs from an Excel table.

        A header with the names of the columns is required. These names should
        match the names expected by the routine (usually the names of the
        parameters, but also similar ones). The program will read every row
        bellow the header until they end or it reaches a NaN.

        Parameters
        ----------
        file : str
            Path to the file containing the shaft parameters.
        sheet_type : str, optional
            Describes the kind of sheet the function should expect:
                Simple: The input table should specify only the number of the materials
                to be used.
                They must be saved prior to calling the method.
                Model: The materials parameters must be passed along with the shaft
                parameters. Each material must have an id number and each shaft must
                reference one of the materials ids.
        sheet_name : int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is
            passed, it is assumed to be the first sheet in the file.

        Returns
        -------
        shaft : list
            A list of shaft objects.
        """
        parameters = read_table_file(
            file, "shaft", sheet_name=sheet_name, sheet_type=sheet_type
        )
        list_of_shafts = []
        if sheet_type == "Model":
            new_materials = {}
            for i in range(0, len(parameters["matno"])):
                new_material = Material(
                    name="shaft_mat_" + str(parameters["matno"][i]),
                    rho=parameters["rhoa"][i],
                    E=parameters["ea"][i],
                    G_s=parameters["ga"][i],
                    color=parameters["color"][i],
                )
                new_materials["shaft_mat_" + str(parameters["matno"][i])] = new_material
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        idl=parameters["idl"][i],
                        odl=parameters["odl"][i],
                        idr=parameters["idr"][i],
                        odr=parameters["odr"][i],
                        material=new_materials[parameters["material"][i]],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                        alpha=parameters["alpha"][i] if "alpha" in parameters else 0,
                        beta=parameters["beta"][i] if "beta" in parameters else 0,
                    )
                )
        elif sheet_type == "Simple":
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        idl=parameters["idl"][i],
                        odl=parameters["odl"][i],
                        idr=parameters["idr"][i],
                        odr=parameters["odr"][i],
                        material=parameters["material"][i],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                        alpha=parameters["alpha"][i] if "alpha" in parameters else 0,
                        beta=parameters["beta"][i] if "beta" in parameters else 0,
                    )
                )
        return list_of_shafts

    @classmethod
    def section(
        cls,
        L,
        ne,
        s_idl,
        s_odl,
        s_idr=None,
        s_odr=None,
        material=None,
        n=None,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        alpha=0,
        beta=0,
    ):
        """Shaft section constructor.

        This method will create a shaft section with length 'L' divided into
        'ne' elements.

        Parameters
        ----------
        i_d : float
            Inner diameter of the section.
        o_d : float
            Outer diameter of the section.
        E : float
            Young's modulus.
        G_s : float
            Shear modulus.
        material : ross.material
            Shaft material.
        n : int, optional
            Element number (coincident with it's first node).
            If not given, it will be set when the rotor is assembled
            according to the element's position in the list supplied to
            the rotor constructor.
        axial_force : float
            Axial force.
        torque : float
            Torque.
        shear_effects : bool
            Determine if shear effects are taken into account.
            Default is False.
        rotary_inertia : bool
            Determine if rotary_inertia effects are taken into account.
            Default is False.
        gyroscopic : bool
            Determine if gyroscopic effects are taken into account.
            Default is False.
        alpha : float, optional
            Mass proportional damping factor.
            Default is zero.
        beta : float, optional
            Stiffness proportional damping factor.
            Default is zero.

        Returns
        -------
        elements : list
            List with the 'ne' shaft elements.

        Examples
        --------
        >>> # shaft material
        >>> from ross.materials import steel
        >>> # shaft inner and outer diameters
        >>> s_idl = 0
        >>> s_odl = 0.01585
        >>> sec = ShaftElement.section(247.65e-3, 4, 0, 15.8e-3, material=steel)
        >>> len(sec)
        4
        >>> sec[0].i_d
        0.0
        """
        if s_idr is None:
            s_idr = s_idl
        if s_odr is None:
            s_odr = s_odl

        le = L / ne

        elements = [
            cls(
                L=le,
                idl=(s_idr - s_idl) * i * le / L + s_idl,
                odl=(s_odr - s_odl) * i * le / L + s_odl,
                idr=(s_idr - s_idl) * (i + 1) * le / L + s_idl,
                odr=(s_odr - s_odl) * (i + 1) * le / L + s_odl,
                material=material,
                n=n,
                shear_effects=shear_effects,
                rotary_inertia=rotary_inertia,
                gyroscopic=gyroscopic,
                alpha=alpha,
                beta=beta,
            )
            for i in range(ne)
        ]

        return elements


class ShaftElement6DoF(ShaftElement):
    r"""A shaft element with 6 degrees of freedom.

    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects. The object can be
    cylindrical or conical and the formulation is based on :cite:`genta1988conical`
    The matrices will be defined considering the following local
    coordinate vector:

    .. math::

        [x_0, y_0, z_0, \alpha_0, \beta_0, \theta_0, x_1, y_1, z_1, \alpha_1, \beta_1, \theta_1]^T
    Where :math:`\x_0` and :math:`\x_1` correspond to horizontal translation,
    :math:`\y_0` and :math:`\y_1` correspond to vertical translation,
    :math:`\z_0` and :math:`\z_1` correspond to axial translation,
    :math:`\alpha_0` and :math:`\alpha_1` correspond to bending on the yz plane,
    :math:`\beta_0` and :math:`\beta_1` correspond to bending on the xz plane
    and :math:`\theta_0` and :math:`\theta_1` correspond to torsion around z axis.

    Parameters
    ----------
    L : float, pint.Quantity
        Element length (m).
    idl : float, pint.Quantity
        Inner diameter of the element at the left position (m).
    odl : float, pint.Quantity
        Outer diameter of the element at the left position (m).
    idr : float, pint.Quantity, optional
        Inner diameter of the element at the right position (m).
        Default is equal to idl value (cylindrical element).
    odr : float, pint.Quantity, optional
        Outer diameter of the element at the right position (m).
        Default is equal to odl value (cylindrical element).
    material : ross.Material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force (N).
    torque : float, optional
        Torque (N*m).
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : str, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'
    alpha : float, optional
        Mass proportional damping factor.
        Default is zero.
    beta : float, optional
        Stiffness proportional damping factor.
        Default is zero.
    tag : str, optional
        Element tag.
        Default is None.

    Returns
    -------
    shaft_element : ross.ShaftElement6DoF
        A 6 degrees of freedom shaft element object.

    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area at half length (m**2).
    A_l : float
        Element section area at left end (m**2).
    A_r : float
        Element section area at right end (m**2).
    beam_cg : float
        Element center of gravity local position (m).
    axial_cg_pos : float
        Element center of gravity global position (m).
        This should be used only after the rotor is built.
        Default is None.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane (m**4).
    phi : float
        Constant that is used according to :cite:`friswell2010dynamics` to
        consider rotary inertia and shear effects. If these are not considered
        :math:`\phi=0`.
    kappa : float
        Shear coefficient for the element.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> from ross.materials import steel
    >>> # Euler-Bernoulli conical element
    >>> Euler_Bernoulli_Element = ShaftElement6DoF(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         idr=0.05, odr=0.15,
    ...                         rotary_inertia=False,
    ...                         shear_effects=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> # Timoshenko cylindrical element. In this case idr and odr are omitted.
    >>> Timoshenko_Element = ShaftElement6DoF(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes
            with the following ordering:
            x_0 - horizontal translation
            y_0 - vertical translation
            z_0 - axial translation
            alpha_0 - rotation around horizontal
            beta_0  - rotation around vertical
            theta_0 - torsion around axial

        Examples
        --------
        >>> sh = ShaftElement6DoF(L=0.5, idl=0.05, odl=0.1, material=steel,
        ...                       rotary_inertia=True, shear_effects=True)
        >>> sh.dof_mapping()["x_0"]
        0
        """
        return dict(
            x_0=0,
            y_0=1,
            z_0=2,
            alpha_0=3,
            beta_0=4,
            theta_0=5,
            x_1=6,
            y_1=7,
            z_1=8,
            alpha_1=9,
            beta_1=10,
            theta_1=11,
        )

    def M(self):
        """Mass matrix for an instance of a 6 DoF shaft element.

        Returns
        -------
        M : np.ndarray
            Mass matrix for the 6 DoF shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement6DoF(0.25, 0, 0.05, material=steel)
        >>> Timoshenko_Element.M().shape
        (12, 12)
        """
        phi = self.phi
        L = self.L
        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        delta = self.delta
        gama = self.gama

        m1 = (
            (468 + 882 * phi + 420 * phi**2)
            + a1 * (108 + 210 * phi + 105 * phi**2)
            + b1 * (38 + 78 * phi + 42 * phi**2)
        )
        m2 = (
            (66 + 115.5 * phi + 52.5 * phi**2)
            + a1 * (21 + 40.5 * phi + 21 * phi**2)
            + b1 * (8.5 + 18 * phi + 10.5 * phi**2)
        ) * L
        m3 = (
            (162 + 378 * phi + 210 * phi**2)
            + a1 * (81 + 189 * phi + 105 * phi**2)
            + b1 * (46 + 111 * phi + 63 * phi**2)
        )
        m4 = (
            (39 + 94.5 * phi + 52.5 * phi**2)
            + a1 * (18 + 40.5 * phi + 21 * phi**2)
            + b1 * (9.5 + 21 * phi + 10.5 * phi**2)
        ) * L
        m5 = (
            (12 + 21 * phi + 10.5 * phi**2)
            + a1 * (4.5 + 9 * phi + 5.25 * phi**2)
            + b1 * (2 + 4.5 * phi + 3 * phi**2)
        ) * L**2
        m6 = (
            (39 + 94.5 * phi + 52.5 * phi**2)
            + a1 * (21 + 54 * phi + 31.5 * phi**2)
            + b1 * (12.5 + 34.5 * phi + 21 * phi**2)
        ) * L
        m7 = (
            (9 + 21 * phi + 10.5 * phi**2)
            + a1 * (4.5 + 10.5 * phi + 5.25 * phi**2)
            + b1 * (2.5 + 6 * phi + 3 * phi**2)
        ) * L**2
        m8 = (
            (468 + 882 * phi + 420 * phi**2)
            + a1 * (360 + 672 * phi + 315 * phi**2)
            + b1 * (290 + 540 * phi + 252 * phi**2)
        )
        m9 = (
            (66 + 115.5 * phi + 52.5 * phi**2)
            + a1 * (45 + 75 * phi + 31.5 * phi**2)
            + b1 * (32.5 + 52.5 * phi + 21 * phi**2)
        ) * L
        m10 = (
            (12 + 21 * phi + 10.5 * phi**2)
            + a1 * (7.5 + 12 * phi + 5.25 * phi**2)
            + b1 * (5 + 7.5 * phi + 3 * phi**2)
        ) * L**2

        # fmt: off
        M = np.array([
            [ m1,   0, 0,   0,  m2, 0,  m3,   0, 0,   0, -m4, 0],
            [  0,  m1, 0, -m2,   0, 0,   0,  m3, 0,  m4,   0, 0],
            [  0,   0, 0,   0,   0, 0,   0,   0, 0,   0,   0, 0],
            [  0, -m2, 0,  m5,   0, 0,   0, -m6, 0, -m7,   0, 0],
            [ m2,   0, 0,   0,  m5, 0,  m6,   0, 0,   0, -m7, 0],
            [  0,   0, 0,   0,   0, 0,   0,   0, 0,   0,   0, 0],
            [ m3,   0, 0,   0,  m6, 0,  m8,   0, 0,   0, -m9, 0],
            [  0,  m3, 0, -m6,   0, 0,   0,  m8, 0,  m9,   0, 0],
            [  0,   0, 0,   0,   0, 0,   0,   0, 0,   0,   0, 0],
            [  0,  m4, 0, -m7,   0, 0,   0,  m9, 0, m10,   0, 0],
            [-m4,   0, 0,   0, -m7, 0, -m9,   0, 0,   0, m10, 0],
            [  0,   0, 0,   0,   0, 0,   0,   0, 0,   0,   0, 0],
        ]) * self.material.rho * self.A_l * L / (1260 * (1 + phi) ** 2)
        # fmt: on

        if self.rotary_inertia:
            # fmt: off
            m11 = 252 + 126 * a2 + 72 * b2 + 45 * gama + 30 * delta
            m12 = (
                21 - 105 * phi
                + a2 * (21 - 42 * phi)
                + b2 * (15 - 21 * phi)
                + gama * (10.5 - 12 * phi)
                + delta * (7.5 - 7.5 * phi)
            ) * L
            m13 = (
                21 - 105 * phi
                - 63 * a2 * phi
                - b2 * (6 + 42 * phi)
                - gama * (7.5 + 30 * phi)
                - delta * (7.5 + 22.5 * phi)
            ) * L
            m14 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (7 - 7 * phi + 17.5 * phi ** 2)
                + b2 * (4 - 7 * phi + 7 * phi ** 2)
                + gama * (2.75 - 5 * phi + 3.5 * phi ** 2)
                + delta * (2 - 3.5 * phi + 2 * phi ** 2)
            ) * L ** 2
            m15 = (
                7 + 35 * phi - 35 * phi ** 2
                + a2 * (3.5 + 17.5 * phi - 17.5 * phi ** 2)
                + b2 * (3 + 10.5 * phi - 10.5 * phi ** 2)
                + gama * (2.75 + 7 * phi - 7 * phi ** 2)
                + delta * (2.5 + 5 * phi - 5 * phi ** 2)
            ) * L ** 2
            m16 = (
                28 + 35 * phi + 70 * phi ** 2
                + a2 * (21 + 42 * phi + 52.5 * phi ** 2)
                + b2 * (18 + 42 * phi + 42 * phi ** 2)
                + gama * (16.25 + 40 * phi + 35 * phi ** 2)
                + delta * (15 + 37.5 * phi + 30 * phi ** 2)
            ) * L ** 2

            Mr = np.array([
                [ m11,    0, 0,    0,  m12, 0, -m11,    0, 0,    0,  m13, 0],
                [   0,  m11, 0, -m12,    0, 0,    0, -m11, 0, -m13,    0, 0],
                [   0,    0, 0,    0,    0, 0,    0,    0, 0,    0,    0, 0],
                [   0, -m12, 0,  m14,    0, 0,    0,  m12, 0, -m15,    0, 0],
                [ m12,    0, 0,    0,  m14, 0, -m12,    0, 0,    0, -m15, 0],
                [   0,    0, 0,    0,    0, 0,    0,    0, 0,    0,    0, 0],
                [-m11,    0, 0,    0, -m12, 0,  m11,    0, 0,    0, -m13, 0],
                [   0, -m11, 0,  m12,    0, 0,    0,  m11, 0,  m13,    0, 0],
                [   0,    0, 0,    0,    0, 0,    0,    0, 0,    0,    0, 0],
                [   0, -m13, 0, -m15,    0, 0,    0,  m13, 0,  m16,    0, 0],
                [ m13,    0, 0,    0, -m15, 0, -m13,    0, 0,    0,  m16, 0],
                [   0,    0, 0,    0,    0, 0,    0,    0, 0,    0,    0, 0],
            ]) * self.material.rho * self.Ie_l / (210 * L * (1 + phi) ** 2)
            # fmt: on
            M += Mr

        # axial motion
        Ae = (self.A_l + self.A_r) / 2
        # fmt: off
        Mam = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]) * self.material.rho * Ae * L / 6
        # fmt: on
        M += Mam

        # torsion
        Je = 2 * (self.Ie_l + self.Ie_r) / 2
        # fmt: off
        Mts = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        ]) * self.material.rho * Je * L / 6
        # fmt: on
        M += Mts

        return M

    def K(self):
        """Stiffness matrix for an instance of a 6 DoF shaft element.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix for the 6 DoF shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement6DoF(0.25, 0, 0.05, material=steel)
        >>> Timoshenko_Element.K().shape
        (12, 12)
        """
        L = self.L
        phi = self.phi
        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        delta = self.delta
        gama = self.gama

        # fmt: off
        k1 = 1260 + 630 * a2 + 504 * b2 + 441 * gama + 396 * delta
        k2 = (
            630
            + 210 * a2
            + 147 * b2
            + 126 * gama
            + 114 * delta
            - phi * (105 * a2 + 105 * b2 + 94.5 * gama + 84 * delta)
        ) * L
        k3 = (
            630
            + 420 * a2
            + 357 * b2
            + 315 * gama
            + 282 * delta
            + phi * (105 * a2 + 105 * b2 + 94.5 * gama + 84 * delta)
        ) * L
        k4 = (
            420 + 210 * phi + 105 * phi ** 2
            + a2 * (105 + 52.5 * phi ** 2)
            + b2 * (56 - 35 * phi + 35 * phi ** 2)
            + gama * (42 - 42 * phi + 26.25 * phi ** 2)
            + delta * (36 - 42 * phi + 21 * phi ** 2)
        ) * L ** 2
        k5 = (
            210 - 210 * phi - 105 * phi ** 2
            + a2 * (105 - 105 * phi - 52.5 * phi ** 2)
            + b2 * (91 - 70 * phi - 35 * phi ** 2)
            + gama * (84 - 52.5 * phi - 26.25 * phi ** 2)
            + delta * (78 - 42 * phi - 21 * phi ** 2)
        ) * L ** 2
        k6 = (
            420 + 210 * phi + 105 * phi ** 2
            + a2 * (315 + 210 * phi + 52.5 * phi ** 2)
            + b2 * (266 + 175 * phi + 35 * phi ** 2)
            + gama * (231 + 147 * phi + 26.25 * phi ** 2)
            + delta * (204 + 126 * phi + 21 * phi ** 2)
        ) * L ** 2
        k7 = 12 + 6 * a1 + 4 * b1
        k8 = (6 + 3 * a1 + 2 * b1) * L
        k9 = (3 + 1.5 * a1 + b1) * L ** 2

        K1 = np.array([
            [ k1,   0,  0,   0,  k2,  0, -k1,   0,  0,   0,  k3,  0],
            [  0,  k1,  0, -k2,   0,  0,   0, -k1,  0, -k3,   0,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [  0, -k2,  0,  k4,   0,  0,   0,  k2,  0,  k5,   0,  0],
            [ k2,   0,  0,   0,  k4,  0, -k2,   0,  0,   0,  k5,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [-k1,   0,  0,   0, -k2,  0,  k1,   0,  0,   0, -k3,  0],
            [  0, -k1,  0,  k2,   0,  0,   0,  k1,  0,  k3,   0,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [  0, -k3,  0,  k5,   0,  0,   0,  k3,  0,  k6,   0,  0],
            [ k3,   0,  0,   0,  k5,  0, -k3,   0,  0,   0,  k6,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
        ]) * self.Ie_l / 105 

        K2 = np.array([
            [ k7,   0,  0,   0,  k8,  0, -k7,   0,  0,   0,  k8,  0],
            [  0,  k7,  0, -k8,   0,  0,   0, -k7,  0, -k8,   0,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [  0, -k8,  0,  k9,   0,  0,   0,  k8,  0,  k9,   0,  0],
            [ k8,   0,  0,   0,  k9,  0, -k8,   0,  0,   0,  k9,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [-k7,   0,  0,   0, -k8,  0,  k7,   0,  0,   0, -k8,  0],
            [  0, -k7,  0,  k8,   0,  0,   0,  k7,  0,  k8,   0,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            [  0, -k8,  0,  k9,   0,  0,   0,  k8,  0,  k9,   0,  0],
            [ k8,   0,  0,   0,  k9,  0, -k8,   0,  0,   0,  k9,  0],
            [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
        ]) * self.Ie * phi * self.A_l / self.A
        # fmt: on

        K = self.material.E * L ** (-3) * (1 + phi) ** (-2) * (K1 + K2)

        # axial force
        k10 = 36 + 60 * phi + 30 * phi**2
        k11 = 3 * L
        k12 = (4 + 5 * phi + 2.5 * phi**2) * L**2
        k13 = (1 + 5 * phi + 2.5 * phi**2) * L**2
        # fmt: off
        Kaf = np.array([
            [ k10,    0,  0,    0,  k11,  0, -k10,    0,  0,    0,  k11,  0],
            [   0,  k10,  0, -k11,    0,  0,    0, -k10,  0, -k11,    0,  0],
            [   0,    0,  0,    0,    0,  0,    0,    0,  0,    0,    0,  0],
            [   0, -k11,  0,  k12,    0,  0,    0,  k11,  0, -k13,    0,  0],
            [ k11,    0,  0,    0,  k12,  0, -k11,    0,  0,    0, -k13,  0],
            [   0,    0,  0,    0,    0,  0,    0,    0,  0,    0,    0,  0],
            [-k10,    0,  0,    0, -k11,  0,  k10,    0,  0,    0, -k11,  0],
            [   0, -k10,  0,  k11,    0,  0,    0,  k10,  0,  k11,    0,  0],
            [   0,    0,  0,    0,    0,  0,    0,    0,  0,    0,    0,  0],
            [   0, -k11,  0, -k13,    0,  0,    0,  k11,  0,  k12,    0,  0],
            [ k11,    0,  0,    0, -k13,  0, -k11,    0,  0,    0,  k12,  0],
            [   0,    0,  0,    0,    0,  0,    0,    0,  0,    0,    0,  0],
        ]) * self.axial_force / (30 * L * (1 + phi) ** 2)
        # fmt: on
        K += Kaf

        # torque
        # fmt: off
        Ktq = np.array([
            [ 0,  0,  0,    1,    0,  0,  0,  0,  0,   -1,    0,  0],
            [ 0,  0,  0,    0,    1,  0,  0,  0,  0,    0,   -1,  0],
            [ 0,  0,  0,    0,    0,  0,  0,  0,  0,    0,    0,  0],
            [ 1,  0,  0,    0, -L/2,  0, -1,  0,  0,    0,  L/2,  0],
            [ 0,  1,  0,  L/2,    0,  0,  0, -1,  0, -L/2,    0,  0],
            [ 0,  0,  0,    0,    0,  0,  0,  0,  0,    0,    0,  0],
            [ 0,  0,  0,   -1,    0,  0,  0,  0,  0,    1,    0,  0],
            [ 0,  0,  0,    0,   -1,  0,  0,  0,  0,    0,    1,  0],
            [ 0,  0,  0,    0,    0,  0,  0,  0,  0,    0,    0,  0],
            [-1,  0,  0,    0, -L/2,  0,  1,  0,  0,    0,  L/2,  0],
            [ 0, -1,  0,  L/2,    0,  0,  0,  1,  0, -L/2,    0,  0],
            [ 0,  0,  0,    0,    0,  0,  0,  0,  0,    0,    0,  0],
        ]) * self.torque / L
        # fmt: on
        K += Ktq

        # axial motion
        Ae = (self.A_l + self.A_r) / 2
        # fmt: off
        Kam = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ]) * self.material.E * Ae / L
        # fmt: on
        K += Kam

        # torsion
        Je = 2 * (self.Ie_l + self.Ie_r) / 2
        # fmt: off
        Kts = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1],
        ]) * self.material.G_s * Je / L
        # fmt: on
        K += Kts

        return K

    def Kst(self):
        """Dynamic stiffness matrix for an instance of a 6 DoF shaft element.

        Stiffness matrix for the 6 DoF shaft element associated with
        the transient motion. It needs to be multiplied by the angular
        acceleration when considered in time dependent analyses.

        Returns
        -------
        Kst : np.ndarray
            Dynamic stiffness matrix for the 6 DoF shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement6DoF(0.25, 0, 0.05, material=steel)
        >>> Timoshenko_Element.Kst().shape
        (12, 12)
        """
        L = self.L
        Ie = (self.Ie_l + self.Ie_r) / 2

        # fmt: off
        Kst = np.array([
            [0,  -36,  0,    3*L,  0,  0,  0,  36,  0,    3*L,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0, -3*L,  0, 4*L**2,  0,  0,  0, 3*L,  0,  -L**2,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0,   36,  0,   -3*L,  0,  0,  0, -36,  0,   -3*L,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
            [0, -3*L,  0,  -L**2,  0,  0,  0, 3*L,  0, 4*L**2,  0,  0],
            [0,    0,  0,      0,  0,  0,  0,   0,  0,      0,  0,  0],
        ]) * self.material.rho * Ie / (15 * L) 
        # fmt: on

        return Kst

    def C(self):
        """Proportional damping matrix for an instance of a 6 DoF shaft element.

        Returns
        -------
        C : np.ndarray
            Proportional damping matrix for the 6 DoF shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft = ShaftElement6DoF(L=0.25, idl=0, odl=0.05, material=steel)
        >>> shaft.C().shape
        (12, 12)
        """
        # proportional damping matrix
        C = self.alpha * self.M() + self.beta * self.K()

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a 6 DoFs shaft element.

        Gyroscopic matrix for the 6 DoF shaft element. It needs to be
        multiplied by the angular velocity when considered in time
        dependent analyses.

        Returns
        -------
        G : np.ndarray
            Gyroscopic matrix for the 6 DoF shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft = ShaftElement6DoF(0.25, 0, 0.05, material=steel)
        >>> shaft.G().shape
        (12, 12)
        """
        if self.gyroscopic:
            phi = self.phi
            L = self.L
            a2 = self.a2
            b2 = self.b2
            delta = self.delta
            gama = self.gama

            # fmt: off
            g1 = 252 + 126 * a2 + 72 * b2 + 45 * gama + 30 * delta
            g2 = (
                21
                - 105 * phi
                + a2 * (21 - 42 * phi)
                + b2 * (15 - 21 * phi)
                + gama * (10.5 - 12 * phi)
                + delta * (7.5 - 7.5 * phi)
            ) * L
            g3 = (
                21
                - 105 * phi
                - 63 * a2 * phi
                - b2 * (6 + 42 * phi)
                - gama * (7.5 + 30 * phi)
                - delta * (7.5 + 22.5 * phi)
            ) * L
            g4 = (
                28
                + 35 * phi
                + 70 * phi ** 2
                + a2 * (7 - 7 * phi + 17.5 * phi ** 2)
                + b2 * (4 - 7 * phi + 7 * phi ** 2)
                + gama * (2.75 - 5 * phi + 3.5 * phi ** 2)
                + delta * (2 - 3.5 * phi + 2 * phi ** 2)
            ) * L ** 2
            g5 = (
                7
                + 35 * phi
                - 35 * phi ** 2
                + a2 * (3.5 + 17.5 * phi - 17.5 * phi ** 2)
                + b2 * (3 + 10.5 * phi - 10.5 * phi ** 2)
                + gama * (2.75 + 7 * phi - 7 * phi * 2)
                + delta * (2.5 + 5 * phi - 5 * phi ** 2)
            ) * L ** 2
            g6 = (
                28
                + 35 * phi
                + 70 * phi**2
                + a2 * (21 + 42 * phi + 52.5 * phi ** 2)
                + b2 * (18 + 42 * phi + 42 * phi ** 2)
                + gama * (16.25 + 40 * phi + 35 * phi ** 2)
                + delta * (15 + 37.5 * phi + 30 * phi ** 2)
            ) * L ** 2

            G = np.array([
                [  0,  g1,  0, -g2,   0,  0,   0, -g1,  0, -g3,   0,  0],
                [-g1,   0,  0,   0, -g2,  0,  g1,   0,  0,   0, -g3,  0],
                [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
                [ g2,   0,  0,   0,  g4,  0, -g2,   0,  0,   0, -g5,  0],
                [  0,  g2,  0, -g4,   0,  0,   0, -g2,  0,  g5,   0,  0],
                [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
                [  0, -g1,  0,  g2,   0,  0,   0,  g1,  0,  g3,   0,  0],
                [ g1,   0,  0,   0,  g2,  0, -g1,   0,  0,   0,  g3,  0],
                [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
                [ g3,   0,  0,   0, -g5,  0, -g3,   0,  0,   0,  g6,  0],
                [  0,  g3,  0,  g5,   0,  0,   0, -g3,  0, -g6,   0,  0],
                [  0,   0,  0,   0,   0,  0,   0,   0,  0,   0,   0,  0],
            ]) * self.material.rho * self.Ie_l * 2 / (210 * L * (1 + phi) ** 2)
            # fmt: on

        else:
            G = np.zeros((12, 12))

        return G
