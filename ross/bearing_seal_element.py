import warnings

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scipy.interpolate as interpolate

from collections import namedtuple
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
                try:
                    if len(self.w) in (2, 3):
                        self.interpolated = interpolate.interp1d(
                            self.w, self.coefficient, kind=len(self.w) - 1
                        )
                except:
                    raise ValueError(
                        "Arguments (coefficients and w)" " must have the same dimension"
                    )
        else:
            self.interpolated = lambda x: np.array(self.coefficient[0])

    def __eq__(self, other):
        if np.allclose(self.__dict__["coefficient"], other.__dict__["coefficient"]):
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
    tag: str, optional
        A tag to name the element
        Default is None.
    n_link: int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
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
        tag=None,
        n_link=None,
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
        self.n_link = n_link
        self.n_l = n
        self.n_r = n

        self.w = np.array(w, dtype=np.float64)
        self.tag = tag
        self.color = "#355d7a"

    def __repr__(self):
        """This function returns a string representation of a bearing element.
        Parameters
        ----------

        Returns
        -------
        A string representation of a bearing object.
        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing # doctest: +ELLIPSIS
        BearingElement(n=0,
         kxx=[...
        """
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n},\n"
            f" kxx={self.kxx}, kxy={self.kxy},\n"
            f" kyx={self.kyx}, kyy={self.kyy},\n"
            f" cxx={self.cxx}, cxy={self.cxy},\n"
            f" cyx={self.cyx}, cyy={self.cyy},\n"
            f" w={self.w}, tag={self.tag!r})"
        )

    def __eq__(self, other):
        """This function allows bearing elements to be compared.
        Parameters
        ----------
        other: object
            The second object to be compared with.

        Returns
        -------
        bool
            True if the comparison is true; False otherwise.
        Examples
        --------
        >>> bearing1 = bearing_example()
        >>> bearing2 = bearing_example()
        >>> bearing1 == bearing2
        True
        """
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
            "n",
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

    def __hash__(self):
        return hash(self.tag)

    def save(self, file_name):
        """Saves a bearing element in a toml format. It works as an auxiliary function of
        the save function in the Rotor class.
        Parameters
        ----------
        file_name: string
            The name of the file the bearing element will be saved in.

        Returns
        -------
        None

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.save('BearingElement.toml')
        """
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
            "tag": self.tag,
        }
        self.dump_data(data, file_name)

    @staticmethod
    def load(file_name="BearingElement"):
        """Loads a list of bearing elements saved in a toml format.
        Parameters
        ----------
        file_name: string
            The name of the file of the bearing element to be loaded.

        Returns
        -------
        A list of bearing elements.

        Examples
        --------
        >>> bearing1 = bearing_example()
        >>> bearing1.save('BearingElement.toml')
        >>> list_of_bearings = BearingElement.load('BearingElement.toml')
        >>> bearing1 == list_of_bearings[0]
        True
        """
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
        """Returns a dictionary with a mapping between degree of freedom and its index.
        Parameters
        ----------

        Returns
        -------
        A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.dof_mapping()
        {'x_0': 0, 'y_0': 1}
        """
        return dict(x_0=0, y_0=1)

    def dof_global_index(self):
        """Get the global index for a element specific degree of freedom."""
        global_index = super().dof_global_index()

        if self.n_link is not None:
            global_index = global_index._asdict()
            global_index[f"x_{self.n_link}"] = 4 * self.n_link
            global_index[f"y_{self.n_link}"] = 4 * self.n_link + 1
            dof_tuple = namedtuple("GlobalIndex", global_index)
            global_index = dof_tuple(**global_index)

        return global_index

    def M(self):
        """Returns the mass matrix.
        Parameters
        ----------

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.M()
        array([[0., 0.],
               [0., 0.]])
        """
        M = np.zeros_like(self.K(0))

        return M

    def K(self, w):
        """Returns the stiffness matrix for a given speed.
        Parameters
        ----------
        w: float
            The speeds (rad/s) to consider.

        Returns
        -------
        A 2x2 matrix of floats containing the kxx, kxy, kyx, and kyy values.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.K(0)
        array([[1000000.,       0.],
               [      0.,  800000.]])
        """
        kxx = self.kxx.interpolated(w)
        kyy = self.kyy.interpolated(w)
        kxy = self.kxy.interpolated(w)
        kyx = self.kyx.interpolated(w)

        K = np.array([[kxx, kxy], [kyx, kyy]])

        if self.n_link is not None:
            # fmt: off
            K = np.vstack((np.hstack([K, -K]),
                           np.hstack([-K, K])))
            # fmt: on

        return K

    def C(self, w):
        """Returns the damping matrix for a given speed.
        Parameters
        ----------
        w: float
            The speeds (rad/s) to consider.

        Returns
        -------
        A 2x2 matrix of floats containing the cxx, cxy, cyx, and cyy values.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.C(0)
        array([[200.,   0.],
               [  0., 150.]])
        """
        cxx = self.cxx.interpolated(w)
        cyy = self.cyy.interpolated(w)
        cxy = self.cxy.interpolated(w)
        cyx = self.cyx.interpolated(w)

        C = np.array([[cxx, cxy], [cyx, cyy]])

        if self.n_link is not None:
            # fmt: off
            C = np.vstack((np.hstack([C, -C]),
                           np.hstack([-C, C])))
            # fmt: on

        return C

    def G(self):
        """Returns the giroscopic matrix.
        Parameters
        ----------

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.G()
        array([[0., 0.],
               [0., 0.]])
        """
        G = np.zeros_like(self.K(0))

        return G

    def patch(self, position, mean, ax, **kwargs):
        """Bearing element patch.
        Patch that will be used to draw the bearing element.

        Parameters
        ----------
        position : tuple
            Position (z, y) in which the patch will be drawn.
        mean : float
            Mean value of shaft element outer diameters
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------

        Examples
        --------
        """
        default_values = dict(lw=1.0, alpha=1.0, c="k")
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        # geometric factors
        zpos, ypos = position
        coils = 6  # number of points to generate spring
        step = mean / 5  # spring step
        a = 2.2  # range(1.8 to 2.8)
        b = 1.4  # range(1.2 to 1.5)
        c = mean / 2.9  # defines width
        d = (coils / a) * step  # modifies damper width
        n = 5  # number of ground lines

        zs0 = zpos - c
        zs1 = zpos + c
        ys0 = ypos + mean / 2

        # plot bottom base
        x_bot = [zpos, zpos, zs0, zs1]
        yl_bot = [ypos, ys0, ys0, ys0]
        yu_bot = [-y for y in yl_bot]
        ax.add_line(mlines.Line2D(x_bot, yl_bot, **kwargs))
        ax.add_line(mlines.Line2D(x_bot, yu_bot, **kwargs))

        # plot top base
        x_top = [zpos, zpos, zs0, zs1]
        yl_top = [
            b * ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
        ]
        yu_top = [-y for y in yl_top]

        ax.add_line(mlines.Line2D(x_top, yl_top, **kwargs))
        ax.add_line(mlines.Line2D(x_top, yu_top, **kwargs))

        # plot ground
        zl_g = [zs0 - step, zs1 + step]
        yl_g = [yl_top[0], yl_top[0]]
        yu_g = [-y for y in yl_g]
        ax.add_line(mlines.Line2D(zl_g, yl_g, **kwargs))
        ax.add_line(mlines.Line2D(zl_g, yu_g, **kwargs))

        step2 = (zl_g[1] - zl_g[0]) / n
        for i in range(n + 1):
            zl_g2 = [(zs0 - step) + step2 * (i), (zs0 - step) + step2 * (i + 1)]
            yl_g2 = [yl_g[0], 1.1 * yl_g[0]]
            yu_g2 = [-y for y in yl_g2]
            ax.add_line(mlines.Line2D(zl_g2, yl_g2, **kwargs))
            ax.add_line(mlines.Line2D(zl_g2, yu_g2, **kwargs))

        # plot spring
        z_spring = np.array([zs0, zs0, zs0, zs0])
        yl_spring = np.array(
            [ys0, ys0 + step, ys0 + coils * step, ys0 + (coils + 1) * step]
        )

        for i in range(coils):
            z_spring = np.insert(z_spring, i + 2, zs0 - (-1) ** i * step)
            yl_spring = np.insert(yl_spring, i + 2, ys0 + (i + 1) * step)
        yu_spring = [-y for y in yl_spring]

        ax.add_line(mlines.Line2D(z_spring, yl_spring, **kwargs))
        ax.add_line(mlines.Line2D(z_spring, yu_spring, **kwargs))

        # plot damper
        z_damper1 = [zs1, zs1]
        yl_damper1 = [ys0, ys0 + d]
        yu_damper1 = [-y for y in yl_damper1]

        ax.add_line(mlines.Line2D(z_damper1, yl_damper1, **kwargs))
        ax.add_line(mlines.Line2D(z_damper1, yu_damper1, **kwargs))

        z_damper2 = [zs1 - step, zs1 - step, zs1 + step, zs1 + step]
        yl_damper2 = [ys0 + 2 * d, ys0 + d, ys0 + d, ys0 + 2 * d]
        yu_damper2 = [-y for y in yl_damper2]

        ax.add_line(mlines.Line2D(z_damper2, yl_damper2, **kwargs))
        ax.add_line(mlines.Line2D(z_damper2, yu_damper2, **kwargs))

        z_damper3 = [zs1 - step, zs1 + step, zs1, zs1]
        yl_damper3 = [
            ys0 + 1.5 * d,
            ys0 + 1.5 * d,
            ys0 + 1.5 * d,
            ys0 + (coils + 1) * step,
        ]
        yu_damper3 = [-y for y in yl_damper3]

        ax.add_line(mlines.Line2D(z_damper3, yl_damper3, **kwargs))
        ax.add_line(mlines.Line2D(z_damper3, yu_damper3, **kwargs))

    def bokeh_patch(self, position, mean, bk_ax, **kwargs):
        """Bearing element patch.
        Patch that will be used to draw the bearing element.

        Parameters
        ----------
        position : tuple
            Position (z, y) in which the patch will be drawn.
        mean : float
            Mean value of shaft element outer diameters
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        """
        default_values = dict(line_width=3, line_alpha=1, color=bokeh_colors[1])
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        # geometric factors
        zpos, ypos = position
        coils = 6  # number of points to generate spring
        step = mean / 5  # spring step
        a = 2.2  # range(1.8 to 2.8)
        b = 1.4  # range(1.2 to 1.5)
        c = mean / 2.9  # defines width
        d = (coils / a) * step  # modifies damper width
        n = 5  # number of ground lines

        zs0 = zpos - c
        zs1 = zpos + c
        ys0 = ypos + mean / 2

        # plot bottom base
        x_bot = [zpos, zpos, zs0, zs1]
        yl_bot = [ypos, ys0, ys0, ys0]
        yu_bot = [-y for y in yl_bot]
        bk_ax.line(x=x_bot, y=yl_bot, **kwargs)
        bk_ax.line(x=x_bot, y=yu_bot, **kwargs)

        # plot top base
        x_top = [zpos, zpos, zs0, zs1]
        yl_top = [
            b * ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
            ys0 + (coils + 1) * step,
        ]
        yu_top = [-y for y in yl_top]

        bk_ax.line(x=x_top, y=yl_top, legend="Bearing", **kwargs)
        bk_ax.line(x=x_top, y=yu_top, legend="Bearing", **kwargs)

        # plot ground
        if self.n_link is None:
            zl_g = [zs0 - step, zs1 + step]
            yl_g = [yl_top[0], yl_top[0]]
            yu_g = [-y for y in yl_g]
            bk_ax.line(x=zl_g, y=yl_g, **kwargs)
            bk_ax.line(x=zl_g, y=yu_g, **kwargs)

            step2 = (zl_g[1] - zl_g[0]) / n
            for i in range(n + 1):
                zl_g2 = [(zs0 - step) + step2 * (i), (zs0 - step) + step2 * (i + 1)]
                yl_g2 = [yl_g[0], 1.1 * yl_g[0]]
                yu_g2 = [-y for y in yl_g2]
                bk_ax.line(x=zl_g2, y=yl_g2, **kwargs)
                bk_ax.line(x=zl_g2, y=yu_g2, **kwargs)

        # plot spring
        z_spring = np.array([zs0, zs0, zs0, zs0])
        yl_spring = np.array(
            [ys0, ys0 + step, ys0 + coils * step, ys0 + (coils + 1) * step]
        )

        for i in range(coils):
            z_spring = np.insert(z_spring, i + 2, zs0 - (-1) ** i * step)
            yl_spring = np.insert(yl_spring, i + 2, ys0 + (i + 1) * step)
        yu_spring = [-y for y in yl_spring]

        bk_ax.line(x=z_spring, y=yl_spring, **kwargs)
        bk_ax.line(x=z_spring, y=yu_spring, **kwargs)

        # plot damper
        z_damper1 = [zs1, zs1]
        yl_damper1 = [ys0, ys0 + d]
        yu_damper1 = [-y for y in yl_damper1]

        bk_ax.line(x=z_damper1, y=yl_damper1, **kwargs)
        bk_ax.line(x=z_damper1, y=yu_damper1, **kwargs)

        z_damper2 = [zs1 - step, zs1 - step, zs1 + step, zs1 + step]
        yl_damper2 = [ys0 + 2 * d, ys0 + d, ys0 + d, ys0 + 2 * d]
        yu_damper2 = [-y for y in yl_damper2]

        bk_ax.line(x=z_damper2, y=yl_damper2, **kwargs)
        bk_ax.line(x=z_damper2, y=yu_damper2, **kwargs)

        z_damper3 = [zs1 - step, zs1 + step, zs1, zs1]
        yl_damper3 = [
            ys0 + 1.5 * d,
            ys0 + 1.5 * d,
            ys0 + 1.5 * d,
            ys0 + (coils + 1) * step,
        ]
        yu_damper3 = [-y for y in yl_damper3]

        bk_ax.line(x=z_damper3, y=yl_damper3, **kwargs)
        bk_ax.line(x=z_damper3, y=yu_damper3, **kwargs)

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
        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/bearing_seal.xls'
        >>> BearingElement.table_to_toml(0, file_path) # doctest: +ELLIPSIS
        {'n': 0, 'kxx': [...
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
        A header with the names of the columns is required. These names should match the names expected by the routine
        (usually the names of the parameters, but also similar ones). The program will read every row bellow the header
        until they end or it reaches a NaN.
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
        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/bearing_seal.xls'
        >>> BearingElement.from_table(0, file_path) # doctest: +ELLIPSIS
        BearingElement(n=0,
         kxx=[...
        """
        parameters = read_table_file(file, "bearing", sheet_name, n)
        return cls(
            n=parameters["n"],
            kxx=parameters["kxx"],
            cxx=parameters["cxx"],
            kyy=parameters["kyy"],
            kxy=parameters["kxy"],
            kyx=parameters["kyx"],
            cyy=parameters["cyy"],
            cxy=parameters["cxy"],
            cyx=parameters["cyx"],
            w=parameters["w"],
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
        Examples
        --------
        >>> nz = 30
        >>> ntheta = 20
        >>> nradius = 11
        >>> length = 0.03
        >>> omega = 157.1
        >>> p_in = 0.
        >>> p_out = 0.
        >>> radius_rotor = 0.0499
        >>> radius_stator = 0.05
        >>> eccentricity = (radius_stator - radius_rotor)*0.2663
        >>> visc = 0.1
        >>> rho = 860.
        >>> BearingElement.from_fluid_flow(0, nz, ntheta, nradius, length, omega, p_in,
        ...                                p_out, radius_rotor, radius_stator,
        ...                                visc, rho, eccentricity=eccentricity) # doctest: +ELLIPSIS
        BearingElement(n=0,
         kxx=[...
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
    """A seal element.
    This class will create a seal element.
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
    seal_leakage: float, optional
        Amount of leakage
    tag : str, optional
        A tag to name the element
        Default is None
    Examples
    --------
    >>> # A seal element located in the first rotor node, with these
    >>> # following stiffness and damping coefficients and speed range from
    >>> # 0 to 200 rad/s
    >>> import ross as rs
    >>> kxx = 1e6
    >>> kyy = 0.8e6
    >>> cxx = 2e2
    >>> cyy = 1.5e2
    >>> w = np.linspace(0, 200, 11)
    >>> seal = rs.SealElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, w=w)
    >>> seal.K(w) # doctest: +ELLIPSIS
    array([[[1000000., 1000000., ...
    >>> seal.C(w) # doctest: +ELLIPSIS
    array([[[200., 200., ...
    """

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
        tag=None,
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
            tag=tag,
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
            [zpos - step, 1.1 * ypos],  # upper
            [zpos - step, 1.1 * ypos + 2 * step],
            [zpos + step, 1.1 * ypos + 2 * step],
            [zpos + step, 1.1 * ypos],
            [zpos - step, 1.1 * ypos],
        ]
        bearing_points_l = [
            [zpos - step, -1.1 * ypos],  # upper
            [zpos - step, -1.1 * ypos - 2 * step],
            [zpos + step, -1.1 * ypos - 2 * step],
            [zpos + step, -1.1 * ypos],
            [zpos - step, -1.1 * ypos],
        ]

        ax.add_patch(mpatches.Polygon(bearing_points_u, color=self.color, picker=True))
        ax.add_patch(mpatches.Polygon(bearing_points_l, color=self.color, picker=True))

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
            top=1.1 * ypos + 2 * step,
            bottom=1.1 * ypos,
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
            top=-1.1 * ypos,
            bottom=-1.1 * ypos - 2 * step,
            left=zpos - step,
            right=zpos + step,
            line_color=bokeh_colors[6],
            line_width=1,
            fill_alpha=0.8,
            fill_color=bokeh_colors[6],
        )


def bearing_example():
    """This function returns an instance of a simple bearing.
    The purpose is to make available a simple model
    so that doctest can be written using it.

    Parameters
    ----------

    Returns
    -------
    An instance of a bearing object.

    Examples
    --------
    >>> bearing = bearing_example()
    >>> bearing.w[0]
    0.0
    """
    kxx = 1e6
    kyy = 0.8e6
    cxx = 2e2
    cyy = 1.5e2
    w = np.linspace(0, 200, 11)
    bearing = BearingElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, w=w)
    return bearing


def seal_example():
    """This function returns an instance of a simple seal.
    The purpose is to make available a simple model
    so that doctest can be written using it.

    Parameters
    ----------

    Returns
    -------
    An instance of a seal object.

    Examples
    --------
    >>> seal = bearing_example()
    >>> seal.w[0]
    0.0
    """
    kxx = 1e6
    kyy = 0.8e6
    cxx = 2e2
    cyy = 1.5e2
    w = np.linspace(0, 200, 11)
    seal = SealElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, w=w)
    return seal
