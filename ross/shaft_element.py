import sys
import warnings
import os
from pathlib import Path

import bokeh.palettes as bp
from bokeh.models import HoverTool, ColumnDataSource
import matplotlib.patches as mpatches
import numpy as np
import toml
import pandas as pd
import xlrd

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
    .. math:: [x_0, y_0, \alpha_0, \beta_0, x_1, y_1, \alpha_1, \beta_1]^T
    Where :math:`\alpha_0` and :math:`\alpha_1` are the bending on the yz plane and
    :math:`\beta_0` and :math:`\beta_1` are the bending on the xz plane.
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

    @classmethod
    def from_table(cls, file, sheet="Simple"):
        """Instantiate one or more shafts using inputs from a table, either excel or csv.
        Parameters
        ----------
        file: str
            Path to the file containing the shaft parameters.
        sheet: str, optional
            Describes the kind of sheet the function should expect:
                Simple: The input table should contain a header with column names equal
                to parameter names in the ShaftElement class, except for
                shear_effects, rotary_inertia, gyroscopic, and shear_method_calc.
                Model: The sheet must follow the expected format. The function will look
                for the parameters according to this format, in the positions they are supposed
                to be found. Headers should be in rows 4 and 20, just before parameters.
        Returns
        -------
        shaft : list
            A list of shaft objects.
        """
        if sheet == "Simple":
            try:
                df = pd.read_excel(file)
            except FileNotFoundError:
                sys.exit(file + " not found.")
            except xlrd.biffh.XLRDError:
                df = pd.read_csv(file)
            try:
                for index, row in df.iterrows():
                    for i in range(0, row.size):
                        if pd.isna(row[i]):
                            warnings.warn(
                                "NaN found in row "
                                + str(index)
                                + " column "
                                + str(i)
                                + ".\n"
                                "It will be replaced with zero."
                            )
                            row[i] = 0
                list_of_shafts = []
                for i, row in df.iterrows():
                    shear_effects = True
                    rotary_inertia = True
                    gyroscopic = True
                    shear_method_calc = "cowper"
                    try:
                        shear_effects = bool(row["shear_effects"])
                    except KeyError:
                        pass
                    try:
                        rotary_inertia = bool(row["rotary_inertia"])
                    except KeyError:
                        pass
                    try:
                        gyroscopic = bool(row["gyroscopic"])
                    except KeyError:
                        pass
                    try:
                        shear_method_calc = row["shear_method_calc"]
                    except KeyError:
                        pass
                    list_of_shafts.append(
                        cls(
                            row.L,
                            row.i_d,
                            row.o_d,
                            Material.use_material(row.material),
                            n=row.n,
                            axial_force=row.axial_force,
                            torque=row.torque,
                            shear_effects=shear_effects,
                            rotary_inertia=rotary_inertia,
                            gyroscopic=gyroscopic,
                            shear_method_calc=shear_method_calc,
                        )
                    )
                return list_of_shafts
            except KeyError:
                sys.exit(
                    "One or more column names did not match the expected. "
                    "Make sure the table header contains the parameters for the "
                    "ShaftElement class. Also, make sure you have a material "
                    "with the given name."
                )
        elif sheet == "Model":
            try:
                df1 = pd.read_excel(file, header=3, nrows=10)
                df2 = pd.read_excel(file, header=19)
                df_unit = pd.read_excel(file, header=16, nrows=2)
            except FileNotFoundError:
                sys.exit(file + " not found.")
            except xlrd.biffh.XLRDError:
                df1 = pd.read_csv(file, header=3, nrows=10)
                df2 = pd.read_csv(file, header=19)
                df_unit = pd.read_csv(file, header=16, nrows=2)
            convert_to_metric = False
            if df_unit["Length"][1] != "meters":
                convert_to_metric = True
            material_name = []
            material_rho = []
            material_e = []
            material_g_s = []
            new_materials = {}
            for index, row in df1.iterrows():
                if not pd.isna(row["matno"]):
                    material_name.append(int(row["matno"]))
                    material_rho.append(row["rhoa"])
                    material_e.append(row["ea"])
                    material_g_s.append(row["ga"])
            if convert_to_metric:
                for i in range(0, len(material_name)):
                    material_rho[i] = material_rho[i] * 27679.904
                    material_e[i] = material_e[i] * 6894.757
                    material_g_s[i] = material_g_s[i] * 6894.757
            for i in range(0, len(material_name)):
                new_material = Material(
                    name="shaft_mat_" + str(material_name[i]),
                    rho=material_rho[i],
                    E=material_e[i],
                    G_s=material_g_s[i],
                )
                new_materials["shaft_mat_" + str(material_name[i])] = new_material
            shaft_l = []
            shaft_i_d = []
            shaft_o_d = []
            shaft_material = []
            shaft_n = []
            shaft_axial_force = []
            for index, row in df2.iterrows():
                shaft_l.append(row["length"])
                shaft_i_d.append(row["id_Left"])
                shaft_o_d.append(row["od_Left"])
                shaft_material.append(int(row["matnum"]))
                shaft_n.append(row["elemnum"] - 1)
                shaft_axial_force.append(row["axial"])
            if convert_to_metric:
                for i in range(0, len(shaft_n)):
                    shaft_l[i] = shaft_l[i] * 0.0254
                    shaft_i_d[i] = shaft_i_d[i] * 0.0254
                    shaft_o_d[i] = shaft_o_d[i] * 0.0254
                    shaft_axial_force[i] = shaft_axial_force[i] * 4.448_221_61
            list_of_shafts = []
            for i in range(0, len(shaft_n)):
                list_of_shafts.append(
                    cls(
                        shaft_l[i],
                        shaft_i_d[i],
                        shaft_o_d[i],
                        new_materials["shaft_mat_" + str(shaft_material[i])],
                        n=shaft_n[i],
                        axial_force=shaft_axial_force[i],
                    )
                )
            return list_of_shafts
        else:
            sys.exit(
                "A valid choice must be given for the parameter 'sheet'. Either 'Simple' or 'Model' "
                "were expected. It was given " + sheet + "."
            )

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def dof_mapping(self):
        return dict(x0=0, y0=1, alpha0=2, beta0=3, x1=4, y1=5, alpha1=6, beta1=7)

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

    def patch(self, position, SR, ax):
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
            legend = "Shaft - Slenderness Ratio < 1.6"
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

    def bokeh_patch(self, position, SR, bk_ax):
        """Shaft element patch.
        Patch that will be used to draw the shaft element.
        Parameters
        ----------
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        Returns
        -------
        """

        if self.n in SR:
            bk_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            bk_color = bokeh_colors[2]
            legend = "Shaft"

        source_u = ColumnDataSource(
            dict(
                top=[self.o_d / 2],
                bottom=[self.i_d / 2],
                left=[position],
                right=[position + self.L],
                elnum=[self.n],
                out_d=[self.o_d],
                in_d=[self.i_d],
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
                elnum=[self.n],
                out_d=[self.o_d],
                in_d=[self.i_d],
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
            legend=legend,
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
        hover.tooltips = [
            ("Element Number :", "@elnum"),
            ("Outer Diameter :", "@out_d"),
            ("Internal Diameter :", "@in_d"),
            ("Element Length :", "@length"),
            ("Material :", "@mat"),
        ]
        hover.mode = "mouse"

        if len(bk_ax.hover) == 0:
            bk_ax.add_tools(hover)

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
