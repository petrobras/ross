import os
from pathlib import Path

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import numpy as np
import toml

import ross
from ross.element import Element
from ross.materials import Material
from ross.materials import steel

__all__ = ["ShaftElement"]

bokeh_colors = bp.RdGy[11]


class ShaftElement(Element):
    r"""A shaft element.
    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:
    .. math:: [x_1, y_1, \alpha_1, \beta_1, x_2, y_2, \alpha_2, \beta_2]^T
    Where :math:`\alpha_1` and :math:`\alpha_2` are the bending on the yz plane and
    :math:`\beta_1` and :math:`\beta_2` are the bending on the xz plane.
    Parameters
    ----------
    L : float
        Element length.
    i_d : float
        Inner diameter of the element.
    o_d : float
        Outer diameter of the element.
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
    shear_method_calc : string
        Determines which shear calculation method the user will adopt
        Default is 'hutchinson'
    Returns
    -------
    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane Ie = pi*r**2/4
    phi : float
        Constant that is used according to [1]_ to consider rotary
        inertia and shear effects. If these are not considered phi=0.
    References
    ----------
    .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
       & AW Lees, published by Cambridge University Press, 2010 pp. 158-166.
    Examples
    --------
    >>> from ross.materials import steel
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> Euler_Bernoulli_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                        shear_effects=False, rotary_inertia=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> Timoshenko_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                   rotary_inertia=True,
    ...                                   shear_effects=True)
    >>> Timoshenko_Element.phi
    0.08795566502463055
    """

    def __init__(
        self,
        L,
        i_d,
        o_d,
        material,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        shear_method_calc="cowper",
    ):

        if type(material) is str:
            os.chdir(Path(os.path.dirname(ross.__file__)))
            self.material = Material.use_material(material)
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

        self.shear_method_calc = shear_method_calc

        self.L = float(L)

        # diameters
        self.i_d = float(i_d)
        self.o_d = float(o_d)
        self.i_d_l = float(i_d)
        self.o_d_l = float(o_d)
        self.i_d_r = float(i_d)
        self.o_d_r = float(o_d)
        self.color = self.material.color

        self.A = np.pi * (o_d ** 2 - i_d ** 2) / 4
        self.volume = self.A * self.L
        self.m = self.material.rho * self.volume
        #  Ie is the second moment of area of the cross section about
        #  the neutral plane Ie = pi*r**2/4
        self.Ie = np.pi * (o_d ** 4 - i_d ** 4) / 64
        phi = 0

        # Slenderness ratio of beam elements (G*A*L^2) / (E*I)
        sld = (self.material.G_s * self.A * self.L ** 2) / (self.material.E * self.Ie)
        self.slenderness_ratio = sld

        # picking a method to calculate the shear coefficient
        # List of avaible methods:
        # hutchinson - kappa as per Hutchinson (2001)
        # cowper - kappa as per Cowper (1996)
        if shear_effects:
            r = i_d / o_d
            r2 = r * r
            r12 = (1 + r2) ** 2
            if shear_method_calc == "hutchinson":
                # Shear coefficient (phi)
                # kappa as per Hutchinson (2001)
                # fmt: off
                kappa = 6*r12*((1+self.material.Poisson)/
                        ((r12*(7 + 12*self.material.Poisson + 4*self.material.Poisson**2) +
                        4*r2*(5 + 6*self.material.Poisson + 2*self.material.Poisson**2))))
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

        self.phi = phi

    def __eq__(self, other):
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def save(self, file_name):
        data = self.load_data(file_name)
        data["ShaftElement"][str(self.n)] = {
            "L": self.L,
            "i_d": self.i_d,
            "o_d": self.o_d,
            "material": self.material.name,
            "n": self.n,
            "axial_force": self.axial_force,
            "torque": self.torque,
            "shear_effects": self.shear_effects,
            "rotary_inertia": self.rotary_inertia,
            "gyroscopic": self.gyroscopic,
            "shear_method_calc": self.shear_method_calc,
        }
        self.dump_data(data, file_name)

    @staticmethod
    def load(file_name="ShaftElement"):
        shaft_elements = []
        with open("ShaftElement.toml", "r") as f:
            shaft_elements_dict = toml.load(f)
            for element in shaft_elements_dict["ShaftElement"]:
                shaft_elements.append(
                    ShaftElement(**shaft_elements_dict["ShaftElement"][element])
                )
        return shaft_elements

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(L={self.L:{0}.{5}}, i_d={self.i_d:{0}.{5}}, "
            f"o_d={self.o_d:{0}.{5}}, material={self.material.name!r}, "
            f"n={self.n})"
        )

    def __str__(self):
        return (
            f"\nElem. N:    {self.n}"
            f"\nLenght:     {self.L:{10}.{5}}"
            f"\nInt. Diam.: {self.i_d:{10}.{5}}"
            f"\nOut. Diam.: {self.o_d:{10}.{5}}"
            f'\n{35*"-"}'
            f"\n{self.material}"
            f"\n"
        )

    def M(self):
        r"""Mass matrix for an instance of a shaft element.
        Returns
        -------
        Mass matrix for the shaft element.
        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        phi = self.phi
        L = self.L

        m01 = 312 + 588 * phi + 280 * phi ** 2
        m02 = (44 + 77 * phi + 35 * phi ** 2) * L
        m03 = 108 + 252 * phi + 140 * phi ** 2
        m04 = -(26 + 63 * phi + 35 * phi ** 2) * L
        m05 = (8 + 14 * phi + 7 * phi ** 2) * L ** 2
        m06 = -(6 + 14 * phi + 7 * phi ** 2) * L ** 2
        # fmt: off
        M = np.array([[m01,    0,    0,  m02,  m03,    0,    0,  m04],
                      [  0,  m01, -m02,    0,    0,  m03, -m04,    0],
                      [  0, -m02,  m05,    0,    0,  m04,  m06,    0],
                      [m02,    0,    0,  m05, -m04,    0,    0,  m06],
                      [m03,    0,    0, -m04,  m01,    0,    0, -m02],
                      [  0,  m03,  m04,    0,    0,  m01,  m02,    0],
                      [  0, -m04,  m06,    0,    0,  m02,  m05,    0],
                      [m04,    0,    0,  m06, -m02,    0,    0,  m05]])
        # fmt: on
        M = self.material.rho * self.A * self.L * M / (840 * (1 + phi) ** 2)

        if self.rotary_inertia:
            ms1 = 36
            ms2 = (3 - 15 * phi) * L
            ms3 = (4 + 5 * phi + 10 * phi ** 2) * L ** 2
            ms4 = (-1 - 5 * phi + 5 * phi ** 2) * L ** 2
            # fmt: off
            Ms = np.array([[ ms1,    0,    0,  ms2, -ms1,    0,    0,  ms2],
                           [   0,  ms1, -ms2,    0,    0, -ms1, -ms2,    0],
                           [   0, -ms2,  ms3,    0,    0,  ms2,  ms4,    0],
                           [ ms2,    0,    0,  ms3, -ms2,    0,    0,  ms4],
                           [-ms1,    0,    0, -ms2,  ms1,    0,    0, -ms2],
                           [   0, -ms1,  ms2,    0,    0,  ms1,  ms2,    0],
                           [   0, -ms2,  ms4,    0,    0,  ms2,  ms3,    0],
                           [ ms2,    0,    0,  ms4, -ms2,    0,    0,  ms3]])
            # fmt: on
            Ms = self.material.rho * self.Ie * Ms / (30 * L * (1 + phi) ** 2)
            M = M + Ms

        return M

    def K(self):
        r"""Stiffness matrix for an instance of a shaft element.
        Returns
        -------
        Stiffness matrix for the shaft element.
        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.K()[:4, :4]/1e6
        array([[45.69644273,  0.        ,  0.        ,  5.71205534],
               [ 0.        , 45.69644273, -5.71205534,  0.        ],
               [ 0.        , -5.71205534,  0.97294287,  0.        ],
               [ 5.71205534,  0.        ,  0.        ,  0.97294287]])
        """
        phi = self.phi
        L = self.L
        # fmt: off
        K = np.array([
            [12,     0,            0,          6*L,  -12,     0,            0,          6*L],
            [0,     12,         -6*L,            0,    0,   -12,         -6*L,            0],
            [0,   -6*L, (4+phi)*L**2,            0,    0,   6*L, (2-phi)*L**2,            0],
            [6*L,    0,            0, (4+phi)*L**2, -6*L,     0,            0, (2-phi)*L**2],
            [-12,    0,            0,         -6*L,   12,     0,            0,         -6*L],
            [0,    -12,          6*L,            0,    0,    12,          6*L,            0],
            [0,   -6*L, (2-phi)*L**2,            0,    0,   6*L, (4+phi)*L**2,            0],
            [6*L,    0,            0, (2-phi)*L**2, -6*L,     0,            0, (4+phi)*L**2]
        ])
        # fmt: on
        K = self.material.E * self.Ie * K / ((1 + phi) * L ** 3)

        return K

    def C(self):
        """Stiffness matrix for an instance of a shaft element.

        Returns
        -------
        C : np.array
           Damping matrix for the shaft element.

        Examples
        --------
        """
        C = np.zeros((8, 8))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.
        Returns
        -------
        Gyroscopic matrix for the shaft element.
        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[-0.        ,  0.01943344, -0.00022681, -0.        ],
               [-0.01943344, -0.        , -0.        , -0.00022681],
               [ 0.00022681, -0.        , -0.        ,  0.0001524 ],
               [-0.        ,  0.00022681, -0.0001524 , -0.        ]])
        """
        phi = self.phi
        L = self.L

        G = np.zeros((8, 8))

        if self.gyroscopic:
            g1 = 36
            g2 = (3 - 15 * phi) * L
            g3 = (4 + 5 * phi + 10 * phi ** 2) * L ** 2
            g4 = (-1 - 5 * phi + 5 * phi ** 2) * L ** 2
            # fmt: off
            G = np.array([[  0, -g1,  g2,   0,   0,  g1,  g2,   0],
                          [ g1,   0,   0,  g2, -g1,   0,   0,  g2],
                          [-g2,   0,   0, -g3,  g2,   0,   0, -g4],
                          [  0, -g2,  g3,   0,   0,  g2,  g4,   0],
                          [  0,  g1, -g2,   0,   0, -g1, -g2,   0],
                          [-g1,   0,   0, -g2,  g1,   0,   0, -g2],
                          [-g2,   0,   0, -g4,  g2,   0,   0, -g3],
                          [  0, -g2,  g4,   0,   0,  g2,  g3,   0]])
            # fmt: on
            G = -self.material.rho * self.Ie * G / (15 * L * (1 + phi) ** 2)

        return G

    def patch(self, position, SR, ax, bk_ax):
        """Shaft element patch.
        Patch that will be used to draw the shaft element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        Returns
        -------
        """
        position_u = [position, self.i_d / 2]  # upper
        position_l = [position, -self.o_d / 2]  # lower
        width = self.L
        height = self.o_d / 2 - self.i_d / 2
        if self.n in SR:
            mpl_color = "yellow"
            bk_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 30"
        else:
            mpl_color = self.color
            bk_color = bokeh_colors[2]
            legend = "Shaft"

        #  matplotlib - plot the upper half of the shaft
        ax.add_patch(
            mpatches.Rectangle(
                position_u,
                width,
                height,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                fc=mpl_color,
                alpha=0.8,
                label=legend,
            )
        )
        #  matplotlib - plot the lower half of the shaft
        ax.add_patch(
            mpatches.Rectangle(
                position_l,
                width,
                height,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                fc=mpl_color,
                alpha=0.8,
            )
        )

        # bokeh plot - plot the shaft
        bk_ax.quad(
            top=self.o_d / 2,
            bottom=self.i_d / 2,
            left=position,
            right=position + self.L,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            legend=legend,
        )
        bk_ax.quad(
            top=-self.i_d / 2,
            bottom=-self.o_d / 2,
            left=position,
            right=position + self.L,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
        )

    @classmethod
    def section(
        cls,
        L,
        ne,
        si_d,
        so_d,
        material,
        n=None,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    ):
        """Shaft section constructor.
        This method will create a shaft section with length 'L'
        divided into 'ne' elements.
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
        Returns
        -------
        elements: list
            List with the 'ne' shaft elements.
        Examples
        --------
        >>> # shaft material
        >>> from ross.materials import steel
        >>> # shaft inner and outer diameters
        >>> si_d = 0
        >>> so_d = 0.01585
        >>> sec = ShaftElement.section(247.65e-3, 4, 0, 15.8e-3, steel)
        >>> len(sec)
        4
        >>> sec[0].i_d
        0.0
        """

        le = L / ne

        elements = [
            cls(le, si_d, so_d, material, n, shear_effects, rotary_inertia, gyroscopic)
            for _ in range(ne)
        ]

        return elements
