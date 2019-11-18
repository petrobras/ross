import os
from pathlib import Path

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import numpy as np
import toml
from bokeh.models import ColumnDataSource, HoverTool

import ross
from ross.element import Element
from ross.materials import Material, steel
from ross.utils import read_table_file

__all__ = ["ShaftElement", "ShaftTaperedElement"]

bokeh_colors = bp.RdGy[11]


class ShaftElement(Element):
    r"""A shaft element.
    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:

    .. math::

        [x_0, y_0, \alpha_0, \beta_0, x_1, y_1, \alpha_1, \beta_1]^T
    Where :math:`\alpha_0` and :math:`\alpha_1` are the bending on the yz plane
    and :math:`\beta_0` and :math:`\beta_1` are the bending on the xz plane.

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
    axial_force : float, optional
        Axial force.
    torque : float, optional
        Torque.
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : string, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'

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
        the neutral plane :math:`Ie = \pi r^2/4`
    phi : float
        Constant that is used according to :cite:`friswell2010dynamics` to
        consider rotary inertia and shear effects. If these are not considered
        :math:`\phi=0`.

    References
    ----------

    .. bibliography:: ../../../docs/refs.bib

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
        tag=None,
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
        self.tag = tag

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

        # Geometric center
        self.beam_cg = L / 2
        self.axial_cg_pos = None

        # Moment of inertia
        self.Im = 0.125 * self.m * (o_d ** 2 + i_d ** 2)

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
        """
        Equality method for comparasions

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
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...        le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft2 = ShaftElement(
        ...        le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 == shaft2
        True
        """
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.tag)

    def save(self, file_name):
        """Save shaft elements to toml file.

        Parameters
        ----------
        file_name : str

        Examples
        --------
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...     le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftElement.toml')
        """
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
        """Load previously saved shaft elements from toml file.

        Parameters
        ----------
        file_name : str, optional

        Examples
        --------
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...     le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftElement.toml')
        >>> shaft2 = ShaftElement.load("ShaftElement.toml")
        >>> shaft2
        [ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=None)]
        """
        shaft_elements = []
        with open("ShaftElement.toml", "r") as f:
            shaft_elements_dict = toml.load(f)
            for element in shaft_elements_dict["ShaftElement"]:
                shaft_elements.append(
                    ShaftElement(**shaft_elements_dict["ShaftElement"][element])
                )
        return shaft_elements

    @classmethod
    def from_table(cls, file, sheet_type="Simple", sheet_name=0):
        """Instantiate one or more shafts using inputs from an Excel table.

        A header with the names of the columns is required. These names should
        match the names expected by the routine (usually the names of the
        parameters, but also similar ones). The program will read every row
        bellow the header until they end or it reaches a NaN.

        Parameters
        ----------
        file: str
            Path to the file containing the shaft parameters.
        sheet_type: str, optional
            Describes the kind of sheet the function should expect:
                Simple: The input table should specify only the number of the materials to be used.
                They must be saved prior to calling the method.
                Model: The materials parameters must be passed along with the shaft parameters. Each
                material must have an id number and each shaft must reference one of the materials ids.
        sheet_name: int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is passed, it is
            assumed to be the first sheet in the file.

        Returns
        -------
        shaft: list
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
                )
                new_materials["shaft_mat_" + str(parameters["matno"][i])] = new_material
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        i_d=parameters["i_d"][i],
                        o_d=parameters["o_d"][i],
                        material=new_materials[parameters["material"][i]],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                    )
                )
        elif sheet_type == "Simple":
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        i_d=parameters["i_d"][i],
                        o_d=parameters["o_d"][i],
                        material=parameters["material"][i],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                    )
                )
        return list_of_shafts

    @property
    def n(self):
        """
        Set the element number as property

        Parameters
        ----------
        Returns
        -------
        n : int
            Element number
        """
        return self._n

    @n.setter
    def n(self, value):
        """
        Method to set a new value for the element number.

        Parameters
        ----------
        value : int
            element number

        Returns
        -------
        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(L=0.25, i_d=0, o_d=0.05, material=steel,
        ...                       rotary_inertia=True, shear_effects=True)
        >>> shaft1.n = 0
        >>> shaft1
        ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=0)
        """
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def dof_mapping(self):
        """
        Method to map the element's degrees of freedom

        Parameters
        ----------
        Returns
        -------
        The numbering of degrees of freedom of each element node.
        """
        return dict(
            x_0=0, y_0=1, alpha_0=2, beta_0=3, x_1=4, y_1=5, alpha_1=6, beta_1=7
        )

    def __repr__(self):
        """This function returns a string representation of a shaft element.

        Parameters
        ----------
        Returns
        -------
        A string representation of a shaft object.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(L=0.25, i_d=0, o_d=0.05, material=steel,
        ...                       rotary_inertia=True, shear_effects=True)
        >>> shaft1
        ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=None)
        """
        return (
            f"{self.__class__.__name__}"
            f"(L={self.L:{0}.{5}}, i_d={self.i_d:{0}.{5}}, "
            f"o_d={self.o_d:{0}.{5}}, material={self.material.name!r}, "
            f"n={self.n})"
        )

    def __str__(self):
        """
        Method to convert object into string
        
        Parameters
        ----------
        Returns
        -------
        The object's parameters translated to strings
        """
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
        M: np.ndarray
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
        K: np.ndarray
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
        """Damping matrix for an instance of a shaft element.

        Returns
        -------
        C: np.ndarray
           Damping matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel)
        >>> # Currently internal damping for the shaft elements is not
        >>> # considered, so the matrix has only zeros.
        >>> Timoshenko_Element.C()[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C = np.zeros((8, 8))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.

        Returns
        -------
        G: np.ndarray
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

    def patch(self, position, check_sld, ax):
        """Shaft element patch.

        Patch that will be used to draw the shaft element.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------

        """
        position_u = [position, self.i_d / 2]  # upper
        position_l = [position, -self.o_d / 2]  # lower
        width = self.L
        height = self.o_d / 2 - self.i_d / 2
        if check_sld is True and self.slenderness_ratio < 1.6:
            mpl_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            mpl_color = self.color
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

    def bokeh_patch(self, position, check_sld, bk_ax):
        """Shaft element patch.

        Patch that will be used to draw the shaft element.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        hover: Bokeh HoverTool
            Bokeh HoverTool axes
        """
        if check_sld is True and self.slenderness_ratio < 1.6:
            bk_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            bk_color = self.material.color
            legend = "Shaft"

        source_u = ColumnDataSource(
            dict(
                top=[self.o_d / 2],
                bottom=[self.i_d / 2],
                left=[position],
                right=[position + self.L],
                sld=[self.slenderness_ratio],
                elnum=[self.n],
                out_d_l=[self.o_d_l],
                out_d_r=[self.o_d_r],
                in_d_l=[self.i_d_l],
                in_d_r=[self.i_d_r],
                length=[self.L],
                mat=[self.material.name],
            )
        )

        source_l = ColumnDataSource(
            dict(
                top=[-self.o_d / 2],
                bottom=[-self.i_d / 2],
                left=[position],
                right=[position + self.L],
                sld=[self.slenderness_ratio],
                elnum=[self.n],
                out_d_l=[self.o_d_l],
                out_d_r=[self.o_d_r],
                in_d_l=[self.i_d_l],
                in_d_r=[self.i_d_r],
                length=[self.L],
                mat=[self.material.name],
            )
        )

        # bokeh plot - plot the shaft
        bk_ax.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            source=source_u,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            legend_label=legend,
            name="u_shaft",
        )
        bk_ax.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            source=source_l,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            name="l_shaft",
        )
        hover = HoverTool(names=["u_shaft", "l_shaft"])

        if check_sld:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Slenderness Ratio :", "@sld"),
            ]
        else:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Left Outer Diameter :", "@out_d_l"),
                ("Left Inner Diameter :", "@in_d_l"),
                ("Right Outer Diameter :", "@out_d_r"),
                ("Right Inner Diameter :", "@in_d_r"),
                ("Element Length :", "@length"),
                ("Material :", "@mat"),
            ]
        hover.mode = "mouse"

        return hover

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


class ShaftTaperedElement(Element):
    r"""A shaft tapered element.
    This class will create a shaft element that may take into
    account shear, rotary inertia and gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:
    .. math:: [x_0, y_0, \alpha_0, \beta_0, x_1, y_1, \alpha_1, \beta_1]^T
    Where :math:`\alpha_0` and :math:`\alpha_1` are the bending on the yz plane
    and :math:`\beta_0` and :math:`\beta_1` are the bending on the xz plane.

    Parameters
    ----------
    L : float
        Element length.
    i_d_l : float
        Inner diameter of the element at the left position.
    i_d_r : float
        Inner diameter of the element at the right position.
    o_d_l : float
        Outer diameter of the element at the left position.
    o_d_r : float
        Outer diameter of the element at the right position.
    material : ross.material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force.
    torque : float, optional
        Torque.
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : string, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'

    Returns
    -------
    A shaft element

    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element mid-section area.
    Ie : float
        Ie is the second moment of area of the mid cross section about
        the neutral plane Ie = pi*r**2/4
    phi : float
        Constant that is used according to [1] to consider rotary
        inertia and shear effects. If these are not considered phi=0.
    References
    ----------
    .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
       & AW Lees, published by Cambridge University Press, 2010 pp. 158-166.
    Examples
    --------
    >>> from ross.materials import steel
    >>> Euler_Bernoulli_Element = ShaftTaperedElement(
    ...                         material=steel, L=0.5, i_d_l=0.05, o_d_l=0.1,
    ...                         i_d_r=0.05, o_d_r=0.15,
    ...                         rotary_inertia=False,
    ...                         shear_effects=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> Timoshenko_Element = ShaftTaperedElement(
    ...                         material=steel, L=0.5, i_d_l=0.05, o_d_l=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    def __init__(
        self,
        material,
        L,
        i_d_l,
        o_d_l,
        i_d_r=None,
        o_d_r=None,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        shear_method_calc="cowper",
        tag=None,
    ):

        if i_d_r is None:
            i_d_r = i_d_l
        if o_d_r is None:
            o_d_r = o_d_l

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

        self.tag = tag

        self.shear_method_calc = shear_method_calc

        self.L = float(L)

        self.o_d = (float(o_d_l) + float(o_d_r)) / 2
        self.i_d = (float(i_d_l) + float(i_d_r)) / 2
        self.i_d_l = float(i_d_l)
        self.o_d_l = float(o_d_l)
        self.i_d_r = float(i_d_r)
        self.o_d_r = float(o_d_r)
        self.color = self.material.color

        # A_l = cross section area from the left side of the element
        A_l = np.pi * (o_d_l ** 2 - i_d_l ** 2) / 4
        self.A_l = A_l

        # Second moment of area of the cross section from the left side
        # of the element
        Ie_l = np.pi * (o_d_l ** 4 - i_d_l ** 4) / 64

        outer = self.o_d_l ** 2 + self.o_d_l * self.o_d_r + self.o_d_r ** 2
        inner = self.i_d_l ** 2 + self.i_d_l * self.i_d_r + self.i_d_r ** 2
        self.volume = np.pi * (self.L / 12) * (outer - inner)
        self.m = self.material.rho * self.volume

        roj = o_d_l / 2
        rij = i_d_l / 2
        rok = o_d_r / 2
        rik = i_d_r / 2

        # geometrical coefficients
        delta_ro = rok - roj
        delta_ri = rik - rij
        a1 = 2 * np.pi * (roj * delta_ro - rij * delta_ri) / A_l
        a2 = np.pi * (roj ** 3 * delta_ro - rij ** 3 * delta_ri) / Ie_l
        b1 = np.pi * (delta_ro ** 2 - delta_ri ** 2) / A_l
        b2 = (
            3
            * np.pi
            * (roj ** 2 * delta_ro ** 2 - rij ** 2 * delta_ri ** 2)
            / (2 * Ie_l)
        )
        gama = np.pi * (roj * delta_ro ** 3 - rij * delta_ri ** 3) / Ie_l
        delta = np.pi * (delta_ro ** 4 - delta_ri ** 4) / (4 * Ie_l)

        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.gama = gama
        self.delta = delta

        # the area is calculated from the cross section located in the middle
        # of the element
        self.A = A_l * (1 + a1 * 0.5 + b1 * 0.5 ** 2)

        # Ie is the second moment of area of the cross section - located in
        # the middle of the element - about the neutral plane
        Ie = Ie_l * (1 + a2 * 0.5 + b2 * 0.5 ** 2 + gama * 0.5 ** 3 + delta * 0.5 ** 4)
        self.Ie = Ie
        self.Ie_l = Ie_l

        phi = 0

        # geometric center
        c1 = (
            roj ** 2
            + 2 * roj * rok
            + 3 * rok ** 2
            - rij ** 2
            - 2 * rij * rik
            - 3 * rik ** 2
        )
        c2 = (roj ** 2 + roj * rok + rok ** 2) - (rij ** 2 + rij * rik + rik ** 2)
        self.beam_cg = L * c1 / (4 * c2)
        self.axial_cg_pos = None

        # Slenderness ratio of beam elements (G*A*L^2) / (E*I)
        sld = (self.material.G_s * self.A * self.L ** 2) / (self.material.E * Ie)
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
            r = ((i_d_l + i_d_r) / 2) / ((o_d_l + o_d_r) / 2)
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

    def __eq__(self, other):
        """
        Equality method for comparasions

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
        >>> from ross.materials import steel
        >>> shaft1 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft2 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 == shaft2
        True
        """
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.tag)

    def save(self, file_name):
        """Save shaft elements to toml file.

        Parameters
        ----------
        file_name : str

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftTaperedElement.toml')
        """
        data = self.load_data(file_name)
        data["ShaftTaperedElement"][str(self.n)] = {
            "L": self.L,
            "i_d_l": self.i_d_l,
            "i_d_r": self.i_d_r,
            "o_d_l": self.o_d_l,
            "o_d_r": self.o_d_r,
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
    def load(file_name="ShaftTaperedElement"):
        """Load previously saved shaft elements from toml file.

        Parameters
        ----------
        file_name : str, optional

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftTaperedElement.toml')
        >>> shaft2 = ShaftTaperedElement.load("ShaftTaperedElement.toml")
        >>> shaft2 # doctest: +ELLIPSIS
        [ShaftTaperedElement(L=0.25, i_d_l=0.0...
        """
        shaft_elements = []
        with open("ShaftTaperedElement.toml", "r") as f:
            shaft_elements_dict = toml.load(f)
            for element in shaft_elements_dict["ShaftTaperedElement"]:
                shaft_elements.append(
                    ShaftTaperedElement(
                        **shaft_elements_dict["ShaftTaperedElement"][element]
                    )
                )
        return shaft_elements

    @property
    def n(self):
        """
        Set the element number as property

        Parameters
        ----------
        Returns
        -------
        n : int
            Element number
        """
        return self._n

    @n.setter
    def n(self, value):
        """
        Method to set a new value for the element number.

        Parameters
        ----------
        value : int
            element number

        Returns
        -------
        Example
        -------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.n = 0
        >>> shaft1 # doctest: +ELLIPSIS
        ShaftTaperedElement(L=0.25, i_d_l=0.0...
        """
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def dof_mapping(self):
        """
        Method to map the element's degrees of freedom

        Parameters
        ----------
        Returns
        -------
        The numbering of degrees of freedom of each element node.
        """
        return dict(
            x_0=0, y_0=1, alpha_0=2, beta_0=3, x_1=4, y_1=5, alpha_1=6, beta_1=7
        )

    def __repr__(self):
        """This function returns a string representation of a shaft element.

        Parameters
        ----------
        Returns
        -------
        A string representation of a shaft object.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftTaperedElement(
        ...        L=0.25, i_d_l=0, i_d_r=0, o_d_l=0.05, o_d_r=0.08,
        ...        material=steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 # doctest: +ELLIPSIS
        ShaftTaperedElement(L=0.25, i_d_l=0.0...
        """
        return (
            f"{self.__class__.__name__}"
            f"(L={self.L:{0}.{5}}, i_d_l={self.i_d_l:{0}.{5}}, "
            f"i_d_r={self.i_d_r:{0}.{5}}, o_d_l={self.o_d_l:{0}.{5}},  "
            f"o_d_r={self.o_d_r:{0}.{5}}, material={self.material.name!r}, "
            f"n={self.n})"
        )

    def __str__(self):
        """
        Method to convert object into string
        
        Parameters
        ----------
        Returns
        -------
        The object's parameters translated to strings
        """
        return (
            f"\nElem. N:    {self.n}"
            f"\nLenght:     {self.L:{10}.{5}}"
            f"\nLeft Int. Diam.: {self.i_d_l:{10}.{5}}"
            f"\nLeft Out. Diam.: {self.o_d_l:{10}.{5}}"
            f"\nRight Int. Diam.: {self.i_d_r:{10}.{5}}"
            f"\nRight Out. Diam.: {self.o_d_r:{10}.{5}}"
            f'\n{35*"-"}'
            f"\n{self.material}"
            f"\n"
        )

    def M(self):
        r"""Mass matrix for an instance of a shaft tapered element.

        Returns
        -------
        Mass matrix for the shaft tapered element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftTaperedElement(
        ...                         L=0.5, i_d_l=0.05, i_d_r=0.05, o_d_l=0.1,
        ...                         o_d_r=0.15, material=steel,
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
            (468 + 882 * phi + 420 * phi ** 2)
            + a1 * (108 + 210 * phi + 105 * phi ** 2)
            + b1 * (38 + 78 * phi + 42 * phi ** 2)
        )
        m2 = (
            (66 + 115.5 * phi + 52.5 * phi ** 2)
            + a1 * (21 + 40.5 * phi + 21 * phi ** 2)
            + b1 * (8.5 + 18 * phi + 10.5 * phi ** 2)
        )
        m3 = (
            (162 + 378 * phi + 210 * phi ** 2)
            + a1 * (81 + 189 * phi + 105 * phi ** 2)
            + b1 * (46 + 111 * phi + 63 * phi ** 2)
        )
        m4 = (
            (39 + 94.5 * phi + 52.5 * phi ** 2)
            + a1 * (18 + 40.5 * phi + 21 * phi ** 2)
            + b1 * (9.5 + 21 * phi + 10.5 * phi ** 2)
        )
        m5 = (
            (12 + 21 * phi + 10.5 * phi ** 2)
            + a1 * (4.5 + 9 * phi + 5.25 * phi ** 2)
            + b1 * (2 + 4.5 * phi + 3 * phi ** 2)
        )
        m6 = (
            (39 + 94.5 * phi + 52.5 * phi ** 2)
            + a1 * (21 + 54 * phi + 31.5 * phi ** 2)
            + b1 * (12.5 + 34.5 * phi + 21 * phi ** 2)
        )
        m7 = (
            (9 + 21 * phi + 10.5 * phi ** 2)
            + a1 * (4.5 + 10.5 * phi + 5.25 * phi ** 2)
            + b1 * (2.5 + 6 * phi + 3 * phi ** 2)
        )
        m8 = (
            (468 + 882 * phi + 420 * phi ** 2)
            + a1 * (360 + 672 * phi + 315 * phi ** 2)
            + b1 * (290 + 540 * phi + 252 * phi ** 2)
        )
        m9 = (
            (66 + 115.5 * phi + 52.5 * phi ** 2)
            + a1 * (45 + 75 * phi + 31.5 * phi ** 2)
            + b1 * (32.5 + 52.5 * phi + 21 * phi ** 2)
        )
        m10 = (
            (12 + 21 * phi + 10.5 * phi ** 2)
            + a1 * (7.5 + 12 * phi + 5.25 * phi ** 2)
            + b1 * (5 + 7.5 * phi + 3 * phi ** 2)
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
        r"""Stiffness matrix for an instance of a shaft tapered element.

        Returns
        -------
        Stiffness matrix for the shaft tapered element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftTaperedElement(
        ...                         L=0.5, i_d_l=0.05, i_d_r=0.05, o_d_l=0.1,
        ...                         o_d_r=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.K()[:4, :4]/1e6
        array([[209.25641985,   0.        ,   0.        ,  38.62129051],
               [  0.        , 209.25641985, -38.62129051,   0.        ],
               [  0.        , -38.62129051,  11.56619973,   0.        ],
               [ 38.62129051,   0.        ,   0.        ,  11.56619973]])
        """
        L = self.L
        phi = self.phi
        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        delta = self.delta
        gama = self.gama
        Ie_l = self.Ie_l

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

        K = self.material.E * Ie_l / (105 * L ** 3 * (1 + phi) ** 2) * (K1 + 105 * phi * K2)
        # fmt: on

        return K

    def C(self):
        """Stiffness matrix for an instance of a shaft tapered element.

        Returns
        -------
        C : np.array
           Damping matrix for the shaft tapered element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftTaperedElement(
        ...                         L=0.5, i_d_l=0.05, i_d_r=0.05, o_d_l=0.1,
        ...                         o_d_r=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.C()[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C = np.zeros((8, 8))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a shaft tapred element.

        Returns
        -------
        Gyroscopic matrix for the shaft tapered element.
 
        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftTaperedElement(
        ...                         L=0.5, i_d_l=0.05, i_d_r=0.05, o_d_l=0.1,
        ...                         o_d_r=0.15, material=steel,
        ...                         rotary_inertia=True,
        ...                         shear_effects=True)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[ 0.        ,  0.30940809, -0.01085902,  0.        ],
               [-0.30940809,  0.        ,  0.        , -0.01085902],
               [ 0.01085902,  0.        ,  0.        ,  0.0067206 ],
               [ 0.        ,  0.01085902, -0.0067206 ,  0.        ]])
        """

        G = np.zeros((8, 8))

        if self.gyroscopic:
            phi = self.phi
            L = self.L
            a1 = self.a1
            a2 = self.a2
            b1 = self.b1
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
                    [    0,    g1,    -L*g2,        0,     0,   -g1,    -L*g3,        0],
                    [  -g1,     0,        0,    -L*g2,    g1,     0,        0,    -L*g3],
                    [ L*g2,     0,        0,  L**2*g4, -L*g2,     0,        0, -L**2*g5],
                    [    0,  L*g2, -L**2*g4,        0,     0, -L*g2,  L**2*g5,        0],
                    [    0,   -g1,     L*g2,        0,     0,    g1,     L*g3,        0],
                    [   g1,     0,        0,     L*g2,   -g1,     0,        0,     L*g3],
                    [ L*g3,     0,        0, -L**2*g5, -L*g3,     0,        0,  L**2*g6],
                    [    0,  L*g3,  L**2*g5,        0,     0, -L*g3, -L**2*g6,        0],
            ])
            # fmt: on
            G = self.material.rho * Ie_l * 2 * G / (210 * L * (1 + phi) ** 2)

        return G

    def patch(self, position, check_sld, ax):
        """Shaft element patch.
        Patch that will be used to draw the shaft element.
        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, color the elements in yellow if slenderness ratio < 1.6
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        """
        if check_sld is True and self.slenderness_ratio < 1.6:
            mpl_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            mpl_color = self.color
            legend = "Shaft"

        shaft_points_u = [
            [position, self.i_d_l / 2],
            [position, self.o_d_l / 2],
            [position + self.L, self.o_d_r / 2],
            [position + self.L, self.i_d_r / 2],
            [position, self.i_d_l / 2],
        ]
        shaft_points_l = [
            [position, -self.i_d_l / 2],
            [position, -self.o_d_l / 2],
            [position + self.L, -self.o_d_r / 2],
            [position + self.L, -self.i_d_r / 2],
            [position, -self.i_d_l / 2],
        ]

        # matplotlib - plot the upper half of the shaft
        ax.add_patch(
            mpatches.Polygon(
                shaft_points_u,
                facecolor=mpl_color,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                alpha=0.8,
                label=legend,
            )
        )
        # matplotlib - plot the lower half of the shaft
        ax.add_patch(
            mpatches.Polygon(
                shaft_points_l,
                facecolor=mpl_color,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                alpha=0.8,
            )
        )

    def bokeh_patch(self, position, check_sld, bk_ax):
        """Shaft element patch.
        Patch that will be used to draw the shaft element.
        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio and color
            the elements in yellow if slenderness ratio < 1.6
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        hover : Bokeh HoverTool
            Bokeh HoverTool axes
        """
        if check_sld is True and self.slenderness_ratio < 1.6:
            bk_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            bk_color = self.material.color
            legend = "Shaft"

        # bokeh plot - plot the shaft
        z_upper = [position, position, position + self.L, position + self.L]
        y_upper = [self.i_d_l / 2, self.o_d_l / 2, self.o_d_r / 2, self.i_d_r / 2]
        z_lower = [position, position, position + self.L, position + self.L]
        y_lower = [-self.i_d_l / 2, -self.o_d_l / 2, -self.o_d_r / 2, -self.i_d_r / 2]

        source = ColumnDataSource(
            dict(
                z_l=[z_lower],
                y_l=[y_lower],
                z_u=[z_upper],
                y_u=[y_upper],
                sld=[self.slenderness_ratio],
                elnum=[self.n],
                out_d_l=[self.o_d_l],
                out_d_r=[self.o_d_r],
                in_d_l=[self.i_d_l],
                in_d_r=[self.i_d_r],
                length=[self.L],
                mat=[self.material.name],
            )
        )

        bk_ax.patches(
            xs="z_u",
            ys="y_u",
            source=source,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            legend_label=legend,
            name="u_shaft",
        )
        bk_ax.patches(
            xs="z_l",
            ys="y_l",
            source=source,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            legend_label=legend,
            name="l_shaft",
        )

        hover = HoverTool(names=["l_shaft", "u_shaft"])
        if check_sld:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Slenderness Ratio :", "@sld"),
            ]
        else:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Left Outer Diameter :", "@out_d_l"),
                ("Left Inner Diameter :", "@in_d_l"),
                ("Right Outer Diameter :", "@out_d_r"),
                ("Right Inner Diameter :", "@in_d_r"),
                ("Element Length :", "@length"),
                ("Material :", "@mat"),
            ]
        hover.mode = "mouse"

        return hover
