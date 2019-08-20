import warnings

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

from ross.utils import read_table_file
from ross.element import Element
from ross.fluid_flow import fluid_flow as flow

__all__ = ["BearingElement", "SealElement"]
bokeh_colors = bp.RdGy[11]


class _Coefficient:
    def __init__(self, coefficient, w=None, interpolated=None):
        if isinstance(coefficient, (int, float)):
            if w is not None and type(w) != float:
                coefficient = [coefficient for _ in range(len(w))]
            else:
                coefficient = [coefficient]

        self.coefficient = coefficient
        self.w = w

        if len(self.coefficient) > 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.interpolated = interpolate.UnivariateSpline(
                        self.w, self.coefficient
                    )
            #  dfitpack.error is not exposed by scipy
            #  so a bare except is used
            except:
                raise ValueError(
                    "Arguments (coefficients and w)" " must have the same dimension"
                )
        else:
            self.interpolated = lambda x: np.array(self.coefficient[0])

    def __eq__(self, other):
        if np.allclose(self.__dict__["coefficient"],other.__dict__["coefficient"]):
            return True
        else:
            return False

    def __repr__(self):
        coef = []
        for i in self.coefficient:
            coef.append("{:.2e}".format(i))
        return f"{coef}"

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        w_range = np.linspace(min(self.w), max(self.w), 30)

        ax.plot(w_range, self.interpolated(w_range), **kwargs)
        ax.set_xlabel("Speed (rad/s)")
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

        return ax


class _Stiffness_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel("Stiffness ($N/m$)")

        return ax


class _Damping_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel("Damping ($Ns/m$)")

        return ax


class BearingElement(Element):
    """A bearing element.
    This class will create a bearing element.
    Parameters can be a constant value or speed dependent.
    For speed dependent parameters, each argument should be passed
    as an array and the correspondent speed values should also be
    passed as an array.
    Values for each parameter will be interpolated for the speed.
    Parameters
    ----------
    n: int
        Node which the bearing will be located in
    kxx: float, array
        Direct stiffness in the x direction.
    cxx: float, array
        Direct damping in the x direction.
    kyy: float, array, optional
        Direct stiffness in the y direction.
        (defaults to kxx)
    cyy: float, array, optional
        Direct damping in the y direction.
        (defaults to cxx)
    kxy: float, array, optional
        Cross coupled stiffness in the x direction.
        (defaults to 0)
    cxy: float, array, optional
        Cross coupled damping in the x direction.
        (defaults to 0)
    kyx: float, array, optional
        Cross coupled stiffness in the y direction.
        (defaults to 0)
    cyx: float, array, optional
        Cross coupled damping in the y direction.
        (defaults to 0)
    w: array, optional
        Array with the speeds (rad/s).
    Examples
    --------
    >>> # A bearing element located in the first rotor node, with these
    >>> # following stiffness and damping coefficients and speed range from
    >>> # 0 to 200 rad/s
    >>> import ross as rs
    >>> kxx = 1e6
    >>> kyy = 0.8e6
    >>> cxx = 2e2
    >>> cyy = 1.5e2
    >>> w = np.linspace(0, 200, 11)
    >>> bearing0 = rs.BearingElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, w=w)
    >>> bearing0.K(w) # doctest: +ELLIPSIS
    array([[[1000000., 1000000., ...
    >>> bearing0.C(w) # doctest: +ELLIPSIS
    array([[[200., 200., ...
    """

    def __init__(
        self, n, kxx, cxx, kyy=None, kxy=0, kyx=0, cyy=None, cxy=0, cyx=0, w=None
    ):

        args = ["kxx", "kyy", "kxy", "kyx", "cxx", "cyy", "cxy", "cyx"]

        # all args to coefficients
        args_dict = locals()
        coefficients = {}

        if kyy is None:
            args_dict["kyy"] = kxx
        if cyy is None:
            args_dict["cyy"] = cxx

        for arg in args:
            if arg[0] == "k":
                coefficients[arg] = _Stiffness_Coefficient(
                    coefficient=args_dict[arg], w=args_dict["w"]
                )
            else:
                coefficients[arg] = _Damping_Coefficient(args_dict[arg], args_dict["w"])

        coefficients_len = [len(v.coefficient) for v in coefficients.values()]

        if w is not None and type(w) != float:
            coefficients_len.append(len(args_dict["w"]))
            if len(set(coefficients_len)) > 1:
                raise ValueError(
                    "Arguments (coefficients and w)" " must have the same dimension"
                )
        else:
            for c in coefficients_len:
                if c != 1:
                    raise ValueError(
                        "Arguments (coefficients and w)" " must have the same dimension"
                    )

        for k, v in coefficients.items():
            setattr(self, k, v)

        self.n = n
        self.n_l = n
        self.n_r = n

        self.w = np.array(w, dtype=np.float64)
        self.color = "#355d7a"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n},\n"
            f" kxx={self.kxx}, kxy={self.kxy},\n"
            f" kyx={self.kyx}, kyy={self.kyy},\n"
            f" cxx={self.cxx}, cxy={self.cxy},\n"
            f" cyx={self.cyx}, cyy={self.cyy},\n"
            f" w={self.w})"
        )

    def __eq__(self, other):
        compared_attributes = [
            "kxx",
            "kyy",
            "kxy",
            "kyx",
            "cxx",
            "cyy",
            "cxy",
            "cyx",
            "w",
        ]
        if isinstance(other, self.__class__):
            return all(
                (
                    np.array(getattr(self, attr)).all()
                    == np.array(getattr(other, attr)).all()
                    for attr in compared_attributes
                )
            )
        return False

    def save(self, file_name):
        data = self.load_data(file_name)
        if type(self.w) == np.ndarray:
            try:
                self.w[0]
                w = list(self.w)
            except IndexError:
                w = None
        data["BearingElement"][str(self.n)] = {
            "n": self.n,
            "kxx": self.kxx.coefficient,
            "cxx": self.cxx.coefficient,
            "kyy": self.kyy.coefficient,
            "kxy": self.kxy.coefficient,
            "kyx": self.kyx.coefficient,
            "cyy": self.cyy.coefficient,
            "cxy": self.cxy.coefficient,
            "cyx": self.cyx.coefficient,
            "w": w,
        }
        self.dump_data(data, file_name)

    @staticmethod
    def load(file_name="BearingElement"):
        bearing_elements = []
        bearing_elements_dict = BearingElement.load_data(
            file_name="BearingElement.toml"
        )
        for element in bearing_elements_dict["BearingElement"]:
            bearing = BearingElement(**bearing_elements_dict["BearingElement"][element])
            bearing.kxx.coefficient = bearing_elements_dict["BearingElement"][element][
                "kxx"
            ]

            bearing.kxy.coefficient = bearing_elements_dict["BearingElement"][element][
                "kxy"
            ]

            bearing.kyx.coefficient = bearing_elements_dict["BearingElement"][element][
                "kyx"
            ]

            bearing.kyy.coefficient = bearing_elements_dict["BearingElement"][element][
                "kyy"
            ]

            bearing.cxx.coefficient = bearing_elements_dict["BearingElement"][element][
                "cxx"
            ]

            bearing.cxy.coefficient = bearing_elements_dict["BearingElement"][element][
                "cxy"
            ]

            bearing.cyx.coefficient = bearing_elements_dict["BearingElement"][element][
                "cyx"
            ]

            bearing.cyy.coefficient = bearing_elements_dict["BearingElement"][element][
                "cyy"
            ]

            bearing_elements.append(bearing)
        return bearing_elements

    def dof_mapping(self):
        return dict(x0=0, y0=1)

    def M(self):
        M = np.zeros((2, 2))

        return M

    def K(self, w):
        kxx = self.kxx.interpolated(w)
        kyy = self.kyy.interpolated(w)
        kxy = self.kxy.interpolated(w)
        kyx = self.kyx.interpolated(w)

        K = np.array([[kxx, kxy], [kyx, kyy]])

        return K

    def C(self, w):
        cxx = self.cxx.interpolated(w)
        cyy = self.cyy.interpolated(w)
        cxy = self.cxy.interpolated(w)
        cyx = self.cyx.interpolated(w)

        C = np.array([[cxx, cxy], [cyx, cyy]])

        return C

    def G(self):
        G = np.zeros((2, 2))

        return G

    def patch(self, position, length, ax):
        """Bearing element patch.
        Patch that will be used to draw the bearing element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position (z, y) in which the patch will be drawn.
        Returns
        -------
        """
        zpos, ypos = position
        step = ypos / 3

        #  node (x pos), outer diam. (y pos)
        bearing_points_u = [
            [zpos - step, ypos],  # upper
            [zpos - step, ypos + 2*step],
            [zpos + step, ypos + 2*step],
            [zpos + step, ypos],
            [zpos - step, ypos],
        ]
        bearing_points_l = [
            [zpos - step, -ypos],  # upper
            [zpos - step, -ypos - 2*step],
            [zpos + step, -ypos - 2*step],
            [zpos + step, -ypos],
            [zpos - step, -ypos],
        ]

        ax.add_patch(
            mpatches.Polygon(bearing_points_u, color=self.color, picker=True)
        )
        ax.add_patch(
            mpatches.Polygon(bearing_points_l, color=self.color, picker=True)
        )

    def bokeh_patch(self, position, length, bk_ax):
        """Bearing element patch.
        Patch that will be used to draw the bearing element.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position (z, y) in which the patch will be drawn.
        length : float
            minimum length of shaft elements

        Returns
        -------
        """
        zpos, ypos = position
        step = ypos / 3

        # bokeh plot - upper bearing visual representation
        bk_ax.quad(
            top=ypos + 2*step,
            bottom=ypos,
            left=zpos - step,
            right=zpos + step,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.8,
            fill_color=bokeh_colors[1],
            legend="Bearing",
        )
        # bokeh plot - lower bearing visual representation
        bk_ax.quad(
            top=-ypos,
            bottom=-ypos - 2*step,
            left=zpos - step,
            right=zpos + step,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.8,
            fill_color=bokeh_colors[1],
        )

    @classmethod
    def table_to_toml(cls, n, file):
        """Convert a table with parameters of a bearing element to a dictionary ready to save
        to a toml file that can be later loaded by ross.
        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        file: str
            Path to the file containing the bearing parameters.
        Returns
        -------
        dict
            A dict that is ready to save to toml and readable by ross.
        """
        b_elem = cls.from_table(n, file)
        data = {
            "n": b_elem.n,
            "kxx": b_elem.kxx.coefficient,
            "cxx": b_elem.cxx.coefficient,
            "kyy": b_elem.kyy.coefficient,
            "kxy": b_elem.kxy.coefficient,
            "kyx": b_elem.kyx.coefficient,
            "cyy": b_elem.cyy.coefficient,
            "cxy": b_elem.cxy.coefficient,
            "cyx": b_elem.cyx.coefficient,
            "w": b_elem.w,
        }
        return data

    @classmethod
    def from_table(cls, n, file, sheet_name=0):
        """Instantiate a bearing using inputs from an Excel table.
        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        file: str
            Path to the file containing the bearing parameters.
        sheet_name: int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is passed, it is
            assumed to be the first sheet in the file.
        Returns
        -------
        A bearing object.
        """
        parameters = read_table_file(file, 'bearing', sheet_name, n)
        return cls(
            n=parameters['n'],
            kxx=parameters['kxx'],
            cxx=parameters['cxx'],
            kyy=parameters['kyy'],
            kxy=parameters['kxy'],
            kyx=parameters['kyx'],
            cyy=parameters['cyy'],
            cxy=parameters['cxy'],
            cyx=parameters['cyx'],
            w=parameters['w'],
        )

    @classmethod
    def from_fluid_flow(
        cls,
        n,
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=None,
        load=None,
    ):
        """Instantiate a bearing using inputs from its fluid flow.
        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        Grid related
        ^^^^^^^^^^^^
        Describes the discretization of the problem
        nz: int
            Number of points along the Z direction (direction of flow).
        ntheta: int
            Number of points along the direction theta. NOTE: ntheta must be odd.
        nradius: int
            Number of points along the direction r.
        length: float
            Length in the Z direction (m).

        Operation conditions
        ^^^^^^^^^^^^^^^^^^^^
        Describes the operation conditions.
        omega: float
            Rotation of the rotor (rad/s).
        p_in: float
            Input Pressure (Pa).
        p_out: float
            Output Pressure (Pa).
        load: float
            Load applied to the rotor (N).

        Geometric data of the problem
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Describes the geometric data of the problem.
        radius_rotor: float
            Rotor radius (m).
        radius_stator: float
            Stator Radius (m).
        eccentricity: float
            Eccentricity (m) is the euclidean distance between rotor and stator centers.
            The center of the stator is in position (0,0).

        Fluid characteristics
        ^^^^^^^^^^^^^^^^^^^^^
        Describes the fluid characteristics.
        visc: float
            Viscosity (Pa.s).
        rho: float
            Fluid density(Kg/m^3).
        Returns
        -------
        A bearing object.
        """
        fluid_flow = flow.PressureMatrix(
            nz,
            ntheta,
            nradius,
            length,
            omega,
            p_in,
            p_out,
            radius_rotor,
            radius_stator,
            visc,
            rho,
            eccentricity=eccentricity,
            load=load,
        )
        k = fluid_flow.get_analytical_damping_matrix()
        c = fluid_flow.get_analytical_stiffness_matrix()
        return cls(
            n,
            kxx=k[0],
            cxx=c[0],
            kyy=k[3],
            kxy=k[1],
            kyx=k[2],
            cyy=c[3],
            cxy=c[1],
            cyx=c[2],
            w=fluid_flow.omega,
        )


class SealElement(BearingElement):
    def __init__(
        self,
        n,
        kxx,
        cxx,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        w=None,
        seal_leakage=None,
    ):
        super().__init__(
            n=n,
            w=w,
            kxx=kxx,
            kxy=kxy,
            kyx=kyx,
            kyy=kyy,
            cxx=cxx,
            cxy=cxy,
            cyx=cyx,
            cyy=cyy,
        )

        self.seal_leakage = seal_leakage
        self.color = "#77ACA2"

    def patch(self, position, length, ax):
        """Seal element patch.
        Patch that will be used to draw the seal element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position in which the patch will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        step = ypos / 3

        #  node (x pos), outer diam. (y pos)
        bearing_points_u = [
            [zpos - step, 1.1*ypos],  # upper
            [zpos - step, 1.1*ypos + 2*step],
            [zpos + step, 1.1*ypos + 2*step],
            [zpos + step, 1.1*ypos],
            [zpos - step, 1.1*ypos],
        ]
        bearing_points_l = [
            [zpos - step, -1.1*ypos],  # upper
            [zpos - step, -1.1*ypos - 2*step],
            [zpos + step, -1.1*ypos - 2*step],
            [zpos + step, -1.1*ypos],
            [zpos - step, -1.1*ypos],
        ]

        ax.add_patch(
            mpatches.Polygon(bearing_points_u, color=self.color, picker=True)
        )
        ax.add_patch(
            mpatches.Polygon(bearing_points_l, color=self.color, picker=True)
        )

    def bokeh_patch(self, position, length, bk_ax):
        """Seal element patch.
        Patch that will be used to draw the seal element.
        Parameters
        ----------
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position in which the patch will be drawn.
        Returns
        -------
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        step = ypos / 3

        # bokeh plot - upper seal visual representation
        bk_ax.quad(
            top=1.1*ypos + 2*step,
            bottom=1.1*ypos,
            left=zpos - step,
            right=zpos + step,
            line_color=bokeh_colors[6],
            line_width=1,
            fill_alpha=0.8,
            fill_color=bokeh_colors[6],
            legend="Seal",
        )
        # bokeh plot - lower seal visual representation
        bk_ax.quad(
            top=-1.1*ypos,
            bottom=-1.1*ypos - 2*step,
            left=zpos - step,
            right=zpos + step,
            line_color=bokeh_colors[6],
            line_width=1,
            fill_alpha=0.8,
            fill_color=bokeh_colors[6],
        )
