"""Bearing Element module.

This module defines the BearingElement classes which will be used to represent the rotor
bearings and seals. There are 7 different classes to represent bearings options,
and 2 element options with 8 or 12 degrees of freedom.
"""
import os
import warnings
from inspect import signature

import numpy as np
import toml
from numpy.polynomial import Polynomial
from plotly import graph_objects as go
from scipy import interpolate as interpolate

from ross.element import Element
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_stiffness_and_damping_coefficients,
)
from ross.units import Q_, check_units
from ross.utils import read_table_file

__all__ = [
    "BearingElement",
    "SealElement",
    "BallBearingElement",
    "RollerBearingElement",
    "BearingFluidFlow",
    "BearingElement6DoF",
    "MagneticBearingElement",
    "CylindricalBearing",
]


class BearingElement(Element):
    """A bearing element.

    This class will create a bearing element.
    Parameters can be a constant value or speed dependent.
    For speed dependent parameters, each argument should be passed
    as an array and the correspondent speed values should also be
    passed as an array.
    Values for each parameter will be_interpolated for the speed.

    Parameters
    ----------
    n : int
        Node which the bearing will be located in
    kxx : float, array, pint.Quantity
        Direct stiffness in the x direction (N/m).
    cxx : float, array, pint.Quantity
        Direct damping in the x direction (N*s/m).
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        (defaults to kxx)
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        (defaults to cxx)
    kxy : float, array, pint.Quantity, optional
        Cross coupled stiffness in the x direction (N/m).
        (defaults to 0)
    cxy : float, array, pint.Quantity, optional
        Cross coupled damping in the x direction (N*s/m).
        (defaults to 0)
    kyx : float, array, pint.Quantity, optional
        Cross coupled stiffness in the y direction (N/m).
        (defaults to 0)
    cyx : float, array, pint.Quantity, optional
        Cross coupled damping in the y direction (N*s/m).
        (defaults to 0)
    frequency : array, pint.Quantity, optional
        Array with the frequencies (rad/s).
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#355d7a' (Cardinal).

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
    >>> frequency = np.linspace(0, 200, 11)
    >>> bearing0 = rs.BearingElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, frequency=frequency)
    >>> bearing0.K(frequency) # doctest: +ELLIPSIS
    array([[[1000000., 1000000., ...
    >>> bearing0.C(frequency) # doctest: +ELLIPSIS
    array([[[200., 200., ...
    """

    @check_units
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
        frequency=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        color="#355d7a",
        **kwargs,
    ):
        if frequency is not None:
            self.frequency = np.array(frequency, dtype=np.float64)
        else:
            self.frequency = frequency
        args = ["kxx", "kyy", "kxy", "kyx", "cxx", "cyy", "cxy", "cyx"]

        # all args to coefficients
        args_dict = locals()

        if kyy is None:
            args_dict["kyy"] = kxx
        if cyy is None:
            args_dict["cyy"] = cxx

        # check coefficients len for consistency
        coefficients_len = []

        for arg in args:
            coefficient, interpolated = self._process_coefficient(args_dict[arg])
            setattr(self, arg, coefficient)
            setattr(self, f"{arg}_interpolated", interpolated)
            coefficients_len.append(len(coefficient))

        if frequency is not None and type(frequency) != float:
            coefficients_len.append(len(args_dict["frequency"]))
            if len(set(coefficients_len)) > 1:
                raise ValueError(
                    "Arguments (coefficients and frequency)"
                    " must have the same dimension"
                )
        else:
            for c in coefficients_len:
                if c != 1:
                    raise ValueError(
                        "Arguments (coefficients and frequency)"
                        " must have the same dimension"
                    )

        self.n = n
        self.n_link = n_link
        self.n_l = n
        self.n_r = n
        self.tag = tag
        self.color = color
        self.scale_factor = scale_factor
        self.dof_global_index = None

    def _process_coefficient(self, coefficient):
        """Helper function used to process the coefficient data."""
        interpolated = None

        if isinstance(coefficient, (int, float)):
            if self.frequency is not None and type(self.frequency) != float:
                coefficient = [coefficient for _ in range(len(self.frequency))]
            else:
                coefficient = [coefficient]

        if len(coefficient) > 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    interpolated = interpolate.UnivariateSpline(
                        self.frequency, coefficient
                    )
            #  dfitpack.error is not exposed by scipy
            #  so a bare except is used
            except:
                try:
                    if len(self.frequency) in (2, 3):
                        interpolated = interpolate.interp1d(
                            self.frequency,
                            coefficient,
                            kind=len(self.frequency) - 1,
                            fill_value="extrapolate",
                        )
                except:
                    raise ValueError(
                        "Arguments (coefficients and frequency)"
                        " must have the same dimension"
                    )
        else:
            interpolated = interpolate.interp1d(
                [0, 1],
                [coefficient[0], coefficient[0]],
                kind="linear",
                fill_value="extrapolate",
            )

        return coefficient, interpolated

    def plot(
        self,
        coefficients=None,
        frequency_units="rad/s",
        stiffness_units="N/m",
        damping_units="N*s/m",
        fig=None,
        **kwargs,
    ):
        """Plot coefficient vs frequency.

        Parameters
        ----------
        coefficients : list, str
            List or str with the coefficients to plot.
        frequency_units : str
            Frequency units.
            Default is rad/s.
        y_units : str
        **kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.

        Example
        -------
        >>> bearing = bearing_example()
        >>> fig = bearing.plot('kxx')
        >>> # fig.show()
        """
        if fig is None:
            fig = go.Figure()

        if isinstance(coefficients, str):
            coefficients = [coefficients]
        # check coefficients consistency
        coefficients_set = set([coeff[0] for coeff in coefficients])
        if len(coefficients_set) > 1:
            raise ValueError("Can only plot stiffness or damping in the same plot.")

        coeff_to_plot = coefficients_set.pop()

        if coeff_to_plot == "k":
            default_units = "N/m"
            y_units = stiffness_units
        else:
            default_units = "N*s/m"
            y_units = damping_units

        _frequency_range = np.linspace(min(self.frequency), max(self.frequency), 30)

        for coeff in coefficients:
            y_value = (
                Q_(
                    getattr(self, f"{coeff}_interpolated")(_frequency_range),
                    default_units,
                )
                .to(y_units)
                .m
            )
            frequency_range = Q_(_frequency_range, "rad/s").to(frequency_units).m

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=y_value,
                    mode="lines",
                    showlegend=True,
                    hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br> Coefficient ({y_units}): %{{y:.3e}}",
                    name=f"{coeff}",
                )
            )

        fig.update_xaxes(title_text=f"Frequency ({frequency_units})")
        fig.update_yaxes(exponentformat="power")
        fig.update_layout(**kwargs)

        return fig

    def __repr__(self):
        """Return a string representation of a bearing element.

        Returns
        -------
        A string representation of a bearing element object.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing # doctest: +ELLIPSIS
        BearingElement(n=0, n_link=None,
         kxx=[...
        """
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n}, n_link={self.n_link},\n"
            f" kxx={self.kxx}, kxy={self.kxy},\n"
            f" kyx={self.kyx}, kyy={self.kyy},\n"
            f" cxx={self.cxx}, cxy={self.cxy},\n"
            f" cyx={self.cyx}, cyy={self.cyy},\n"
            f" frequency={self.frequency}, tag={self.tag!r})"
        )

    def __eq__(self, other):
        """Equality method for comparasions.

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
            "frequency",
        ]
        if isinstance(other, self.__class__):
            init_args = []
            for arg in signature(self.__init__).parameters:
                if arg not in ["kwargs"]:
                    init_args.append(arg)

            init_args_comparison = []
            for arg in init_args:
                comparison = getattr(self, arg) == getattr(other, arg)
                try:
                    comparison = all(comparison)
                except TypeError:
                    pass

                init_args_comparison.append(comparison)

            init_args_comparison = all(init_args_comparison)
            attributes_comparison = all(
                (
                    (
                        np.array(getattr(self, attr)) == np.array(getattr(other, attr))
                    ).all()
                    for attr in compared_attributes
                )
            )

            return init_args_comparison and attributes_comparison
        return False

    def __hash__(self):
        return hash(self.tag)

    def save(self, file):
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        # save initialization args and coefficients
        args = list(signature(self.__init__).parameters)
        args += [
            "kxx",
            "kyy",
            "kxy",
            "kyx",
            "cxx",
            "cyy",
            "cxy",
            "cyx",
        ]
        brg_data = {}
        for arg in args:
            if arg not in ["kwargs"]:
                brg_data[arg] = self.__dict__[arg]

        # change np.array to lists so that we can save in .toml as list(floats)
        for k, v in brg_data.items():
            if isinstance(v, np.generic):
                brg_data[k] = brg_data[k].item()
            elif isinstance(v, np.ndarray):
                brg_data[k] = brg_data[k].tolist()
            # case for a container with np.float (e.g. list(np.float))
            else:
                try:
                    brg_data[k] = [i.item() for i in brg_data[k]]
                except (TypeError, AttributeError):
                    pass

        data[f"{self.__class__.__name__}_{self.tag}"] = brg_data

        with open(file, "w") as f:
            toml.dump(data, f)

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its
        index.

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

        >>> bearing = bearing_example()
        >>> bearing.dof_mapping()
        {'x_0': 0, 'y_0': 1}
        """
        return dict(x_0=0, y_0=1)

    def M(self):
        """Mass matrix for an instance of a bearing element.

        This method returns the mass matrix for an instance of a bearing
        element.

        Returns
        -------
        M : np.ndarray
            Mass matrix (kg).

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.M()
        array([[0., 0.],
               [0., 0.]])
        """
        M = np.zeros_like(self.K(0))

        return M

    @check_units
    def K(self, frequency):
        """Stiffness matrix for an instance of a bearing element.

        This method returns the stiffness matrix for an instance of a bearing
        element.

        Parameters
        ----------
        frequency : float
            The excitation frequency (rad/s).

        Returns
        -------
        K : np.ndarray
            A 2x2 matrix of floats containing the kxx, kxy, kyx, and kyy values.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.K(0)
        array([[1000000.,       0.],
               [      0.,  800000.]])
        """
        kxx = self.kxx_interpolated(frequency)
        kyy = self.kyy_interpolated(frequency)
        kxy = self.kxy_interpolated(frequency)
        kyx = self.kyx_interpolated(frequency)

        K = np.array([[kxx, kxy], [kyx, kyy]])

        if self.n_link is not None:
            # fmt: off
            K = np.vstack((np.hstack([K, -K]),
                           np.hstack([-K, K])))
            # fmt: on

        return K

    @check_units
    def C(self, frequency):
        """Damping matrix for an instance of a bearing element.

        This method returns the damping matrix for an instance of a bearing
        element.

        Parameters
        ----------
        frequency : float
            The excitation frequency (rad/s).

        Returns
        -------
        C : np.ndarray
            A 2x2 matrix of floats containing the cxx, cxy, cyx, and cyy values (N*s/m).

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.C(0)
        array([[200.,   0.],
               [  0., 150.]])
        """
        cxx = self.cxx_interpolated(frequency)
        cyy = self.cyy_interpolated(frequency)
        cxy = self.cxy_interpolated(frequency)
        cyx = self.cyx_interpolated(frequency)

        C = np.array([[cxx, cxy], [cyx, cyy]])

        if self.n_link is not None:
            # fmt: off
            C = np.vstack((np.hstack([C, -C]),
                           np.hstack([-C, C])))
            # fmt: on

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a bearing element.

        This method returns the mass matrix for an instance of a bearing
        element.

        Returns
        -------
        G : np.ndarray
            A 2x2 matrix of floats.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.G()
        array([[0., 0.],
               [0., 0.]])
        """
        G = np.zeros_like(self.K(0))

        return G

    def _patch(self, position, fig):
        """Bearing element patch.

        Patch that will be used to draw the bearing element using Plotly library.

        Parameters
        ----------
        position : tuple
            Position (z, y_low, y_upp) in which the patch will be drawn.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        default_values = dict(
            mode="lines",
            line=dict(width=1, color=self.color),
            name=self.tag,
            legendgroup="bearings",
            showlegend=False,
            hoverinfo="none",
        )

        # geometric factors
        zpos, ypos, ypos_s = position

        icon_h = ypos_s - ypos  # bearing icon height
        icon_w = icon_h / 2.0  # bearing icon width
        coils = 6  # number of points to generate spring
        n = 5  # number of ground lines
        step = icon_w / (coils + 1)  # spring step

        zs0 = zpos - (icon_w / 3.5)
        zs1 = zpos + (icon_w / 3.5)
        ys0 = ypos + 0.25 * icon_h

        # plot bottom base
        x_bot = [zpos, zpos, zs0, zs1]
        yl_bot = [ypos, ys0, ys0, ys0]
        yu_bot = [-y for y in yl_bot]

        fig.add_trace(go.Scatter(x=x_bot, y=yl_bot, **default_values))
        fig.add_trace(go.Scatter(x=x_bot, y=yu_bot, **default_values))

        # plot top base
        x_top = [zpos, zpos, zs0, zs1]
        yl_top = [
            ypos + icon_h,
            ypos + 0.75 * icon_h,
            ypos + 0.75 * icon_h,
            ypos + 0.75 * icon_h,
        ]
        yu_top = [-y for y in yl_top]
        fig.add_trace(go.Scatter(x=x_top, y=yl_top, **default_values))
        fig.add_trace(go.Scatter(x=x_top, y=yu_top, **default_values))

        # plot ground
        if self.n_link is None:
            zl_g = [zs0 - step, zs1 + step]
            yl_g = [yl_top[0], yl_top[0]]
            yu_g = [-y for y in yl_g]
            fig.add_trace(go.Scatter(x=zl_g, y=yl_g, **default_values))
            fig.add_trace(go.Scatter(x=zl_g, y=yu_g, **default_values))

            step2 = (zl_g[1] - zl_g[0]) / n
            for i in range(n + 1):
                zl_g2 = [(zs0 - step) + step2 * (i), (zs0 - step) + step2 * (i + 1)]
                yl_g2 = [yl_g[0], 1.1 * yl_g[0]]
                yu_g2 = [-y for y in yl_g2]
                fig.add_trace(go.Scatter(x=zl_g2, y=yl_g2, **default_values))
                fig.add_trace(go.Scatter(x=zl_g2, y=yu_g2, **default_values))

        # plot spring
        z_spring = np.array([zs0, zs0, zs0, zs0])
        yl_spring = np.array([ys0, ys0 + step, ys0 + icon_w - step, ys0 + icon_w])

        for i in range(coils):
            z_spring = np.insert(z_spring, i + 2, zs0 - (-1) ** i * step)
            yl_spring = np.insert(yl_spring, i + 2, ys0 + (i + 1) * step)
        yu_spring = [-y for y in yl_spring]

        fig.add_trace(go.Scatter(x=z_spring, y=yl_spring, **default_values))
        fig.add_trace(go.Scatter(x=z_spring, y=yu_spring, **default_values))

        # plot damper - base
        z_damper1 = [zs1, zs1]
        yl_damper1 = [ys0, ys0 + 2 * step]
        yu_damper1 = [-y for y in yl_damper1]
        fig.add_trace(go.Scatter(x=z_damper1, y=yl_damper1, **default_values))
        fig.add_trace(go.Scatter(x=z_damper1, y=yu_damper1, **default_values))

        # plot damper - center
        z_damper2 = [zs1 - 2 * step, zs1 - 2 * step, zs1 + 2 * step, zs1 + 2 * step]
        yl_damper2 = [ys0 + 5 * step, ys0 + 2 * step, ys0 + 2 * step, ys0 + 5 * step]
        yu_damper2 = [-y for y in yl_damper2]
        fig.add_trace(go.Scatter(x=z_damper2, y=yl_damper2, **default_values))
        fig.add_trace(go.Scatter(x=z_damper2, y=yu_damper2, **default_values))

        # plot damper - top
        z_damper3 = [z_damper2[0], z_damper2[2], zs1, zs1]
        yl_damper3 = [
            ys0 + 4 * step,
            ys0 + 4 * step,
            ys0 + 4 * step,
            ypos + 1.5 * icon_w,
        ]
        yu_damper3 = [-y for y in yl_damper3]

        fig.add_trace(go.Scatter(x=z_damper3, y=yl_damper3, **default_values))
        fig.add_trace(go.Scatter(x=z_damper3, y=yu_damper3, **default_values))

        return fig

    @classmethod
    def table_to_toml(cls, n, file):
        """Convert bearing parameters to toml.

        Convert a table with parameters of a bearing element to a dictionary ready to
        save to a toml file that can be later loaded by ross.

        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        file : str
            Path to the file containing the bearing parameters.

        Returns
        -------
        data : dict
            A dict that is ready to save to toml and readable by ross.

        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/bearing_seal_si.xls'
        >>> BearingElement.table_to_toml(0, file_path) # doctest: +ELLIPSIS
        {'n': 0, 'kxx': array([...
        """
        b_elem = cls.from_table(n, file)
        data = {
            "n": b_elem.n,
            "kxx": b_elem.kxx,
            "cxx": b_elem.cxx,
            "kyy": b_elem.kyy,
            "kxy": b_elem.kxy,
            "kyx": b_elem.kyx,
            "cyy": b_elem.cyy,
            "cxy": b_elem.cxy,
            "cyx": b_elem.cyx,
            "frequency": b_elem.frequency,
        }
        return data

    @classmethod
    def from_table(
        cls,
        n,
        file,
        sheet_name=0,
        tag=None,
        n_link=None,
        scale_factor=1,
        color="#355d7a",
    ):
        """Instantiate a bearing using inputs from an Excel table.

        A header with the names of the columns is required. These names should match the
        names expected by the routine (usually the names of the parameters, but also
        similar ones). The program will read every row bellow the header
        until they end or it reaches a NaN.

        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        file : str
            Path to the file containing the bearing parameters.
        sheet_name : int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is
            passed, it is assumed to be the first sheet in the file.
        tag : str, optional
            A tag to name the element.
            Default is None.
        n_link : int, optional
            Node to which the bearing will connect. If None the bearing is connected to
            ground.
            Default is None.
        scale_factor : float, optional
            The scale factor is used to scale the bearing drawing.
            Default is 1.
        color : str, optional
            A color to be used when the element is represented.
            Default is '#355d7a' (Cardinal).

        Returns
        -------
        bearing : rs.BearingElement
            A bearing object.

        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/bearing_seal_si.xls'
        >>> BearingElement.from_table(0, file_path, n_link=1) # doctest: +ELLIPSIS
        BearingElement(n=0, n_link=1,
         kxx=[1.379...
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
            frequency=parameters["frequency"],
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
            color=color,
        )


class BearingFluidFlow(BearingElement):
    """Instantiate a bearing using inputs from its fluid flow.

    This method always creates elements with frequency-dependent coefficients.
    It calculates a set of coefficients for each frequency value appendend to
    "omega".

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
    omega: list
        List of frequencies (rad/s) used to calculate the coefficients.
        If the length is greater than 1, an array of coefficients is returned.
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

    Others
    ^^^^^^
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    Returns
    -------
    bearing: rs.BearingElement
        A bearing object.

    Examples
    --------
    >>> nz = 30
    >>> ntheta = 20
    >>> length = 0.03
    >>> omega = [157.1]
    >>> p_in = 0.
    >>> p_out = 0.
    >>> radius_rotor = 0.0499
    >>> radius_stator = 0.05
    >>> load = 525
    >>> visc = 0.1
    >>> rho = 860.
    >>> BearingFluidFlow(0, nz, ntheta, length, omega, p_in,
    ...                  p_out, radius_rotor, radius_stator,
    ...                  visc, rho, load=load) # doctest: +ELLIPSIS
    BearingFluidFlow(n=0, n_link=None,
     kxx=[145...
    """

    def __init__(
        self,
        n,
        nz,
        ntheta,
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
        tag=None,
        n_link=None,
        scale_factor=1.0,
        color="#355d7a",
    ):
        K = np.zeros((4, len(omega)))
        C = np.zeros((4, len(omega)))

        for i, w in enumerate(omega):
            fluid_flow = flow.FluidFlow(
                nz,
                ntheta,
                length,
                w,
                p_in,
                p_out,
                radius_rotor,
                radius_stator,
                visc,
                rho,
                eccentricity=eccentricity,
                load=load,
            )
            K[:, i], C[:, i] = calculate_stiffness_and_damping_coefficients(fluid_flow)

        super().__init__(
            n,
            kxx=K[0],
            kxy=K[1],
            kyx=K[2],
            kyy=K[3],
            cxx=C[0],
            cxy=C[1],
            cyx=C[2],
            cyy=C[3],
            frequency=omega,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
            color=color,
        )


class SealElement(BearingElement):
    """A seal element.

    This class will create a seal element.
    Parameters can be a constant value or speed dependent.
    For speed dependent parameters, each argument should be passed
    as an array and the correspondent speed values should also be
    passed as an array.
    Values for each parameter will be_interpolated for the speed.

    SealElement objects are handled differently in the Rotor class, even though it
    inherits from BearingElement class. Seal elements are not considered in static
    analysis, i.e., it does not add reaction forces (only bearings support the rotor).
    In stability level 1 analysis, seal elements are removed temporarily from the model,
    so that the cross coupled coefficients are calculated and replace the seals from
    the rotor model.
    SealElement data is stored in an individual data frame, separate from other
    bearing elements.

    Notes
    -----
    SealElement class is strongly recommended to represent seals.
    Avoid using BearingElement class for this purpose.

    Parameters
    ----------
    n: int
        Node which the bearing will be located in
    kxx : float, array, pint.Quantity
        Direct stiffness in the x direction (N/m).
    cxx : float, array, pint.Quantity
        Direct damping in the x direction (N*s/m).
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        (defaults to kxx)
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        (defaults to cxx)
    kxy : float, array, pint.Quantity, optional
        Cross coupled stiffness in the x direction (N/m).
        (defaults to 0)
    cxy : float, array, pint.Quantity, optional
        Cross coupled damping in the x direction (N*s/m).
        (defaults to 0)
    kyx : float, array, pint.Quantity, optional
        Cross coupled stiffness in the y direction (N/m).
        (defaults to 0)
    cyx : float, array, pint.Quantity, optional
        Cross coupled damping in the y direction (N*s/m).
        (defaults to 0)
    frequency : array, pint.Quantity, optional
        Array with the frequencies (rad/s).
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is "#77ACA2".

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
    >>> frequency = np.linspace(0, 200, 11)
    >>> seal = rs.SealElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, frequency=frequency)
    >>> seal.K(frequency) # doctest: +ELLIPSIS
    array([[[1000000., 1000000., ...
    >>> seal.C(frequency) # doctest: +ELLIPSIS
    array([[[200., 200., ...
    """

    @check_units
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
        frequency=None,
        seal_leakage=None,
        tag=None,
        n_link=None,
        scale_factor=1.0,
        color="#77ACA2",
        **kwargs,
    ):
        # make seals with half the bearing size as a default
        seal_scale_factor = scale_factor / 2
        super().__init__(
            n=n,
            frequency=frequency,
            kxx=kxx,
            kxy=kxy,
            kyx=kyx,
            kyy=kyy,
            cxx=cxx,
            cxy=cxy,
            cyx=cyx,
            cyy=cyy,
            tag=tag,
            n_link=n_link,
            scale_factor=seal_scale_factor,
            color=color,
        )

        self.seal_leakage = seal_leakage


class BallBearingElement(BearingElement):
    """A bearing element for ball bearings.

    This class will create a bearing element based on some geometric and
    constructive parameters of ball bearings. The main difference is that
    cross-coupling stiffness and damping are not modeled in this case.

    The theory used to calculate the stiffness coeficients is based on
    :cite:`friswell2010dynamics` (pages 182-185). Damping is low in rolling-element
    bearings and the direct damping coefficient is typically in the range of
    (0.25 ~ 2.5) x 10e-5 x Kxx (or Kyy).

    Parameters
    ----------
    n : int
        Node which the bearing will be located in.
    n_balls : float
        Number of steel spheres in the bearing.
    d_balls : float
        Diameter of the steel sphere.
    fs : float,optional
        Static bearing loading force.
    alpha : float, optional
        Contact angle between the steel sphere and the inner / outer raceway.
    cxx : float, optional
        Direct stiffness in the x direction.
        Default is 1.25*10e-5 * kxx.
    cyy : float, optional
        Direct damping in the y direction.
        Default is 1.25*10e-5 * kyy.
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> n = 0
    >>> n_balls= 8
    >>> d_balls = 0.03
    >>> fs = 500.0
    >>> alpha = np.pi / 6
    >>> tag = "ballbearing"
    >>> bearing = BallBearingElement(n=n, n_balls=n_balls, d_balls=d_balls,
    ...                              fs=fs, alpha=alpha, tag=tag)
    >>> bearing.K(0)
    array([[4.64168838e+07, 0.00000000e+00],
           [0.00000000e+00, 1.00906269e+08]])
    """

    def __init__(
        self,
        n,
        n_balls,
        d_balls,
        fs,
        alpha,
        cxx=None,
        cyy=None,
        tag=None,
        n_link=None,
        scale_factor=1,
    ):

        Kb = 13.0e6
        kyy = (
            Kb
            * n_balls ** (2.0 / 3)
            * d_balls ** (1.0 / 3)
            * fs ** (1.0 / 3)
            * (np.cos(alpha)) ** (5.0 / 3)
        )

        nb = [8, 12, 16]
        ratio = [0.46, 0.64, 0.73]
        dict_ratio = dict(zip(nb, ratio))

        if n_balls in dict_ratio.keys():
            kxx = dict_ratio[n_balls] * kyy
        else:
            f = interpolate.interp1d(nb, ratio, "quadratic", fill_value="extrapolate")
            kxx = f(n_balls) * kyy

        if cxx is None:
            cxx = 1.25e-5 * kxx
        if cyy is None:
            cyy = 1.25e-5 * kyy

        super().__init__(
            n=n,
            frequency=None,
            kxx=kxx,
            kxy=0.0,
            kyx=0.0,
            kyy=kyy,
            cxx=cxx,
            cxy=0.0,
            cyx=0.0,
            cyy=cyy,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
        )

        self.color = "#77ACA2"


class RollerBearingElement(BearingElement):
    """A bearing element for roller bearings.

    This class will create a bearing element based on some geometric and
    constructive parameters of roller bearings. The main difference is that
    cross-coupling stiffness and damping are not modeled in this case.

    The theory used to calculate the stiffness coeficients is based on
    :cite:`friswell2010dynamics` (pages 182-185). Damping is low in rolling-element
    bearings and the direct damping coefficient is typically in the range of
    (0.25 ~ 2.5) x 10e-5 x Kxx (or Kyy).

    Parameters
    ----------
    n : int
        Node which the bearing will be located in.
    n_rollers : float
        Number of steel spheres in the bearing.
    l_rollers : float
        Length of the steel rollers.
    fs : float, optional
        Static bearing loading force.
    alpha : float, optional
        Contact angle between the steel sphere and the inner / outer raceway.
    cxx : float, optional
        Direct stiffness in the x direction.
        Default is 1.25*10e-5 * kxx.
    cyy : float, optional
        Direct damping in the y direction.
        Default is 1.25*10e-5 * kyy.
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> n = 0
    >>> n_rollers = 8
    >>> l_rollers = 0.03
    >>> fs = 500.0
    >>> alpha = np.pi / 6
    >>> tag = "rollerbearing"
    >>> bearing = RollerBearingElement(n=n, n_rollers=n_rollers, l_rollers=l_rollers,
    ...                            fs=fs, alpha=alpha, tag=tag)
    >>> bearing.K(0)
    array([[2.72821927e+08, 0.00000000e+00],
           [0.00000000e+00, 5.56779444e+08]])
    """

    def __init__(
        self,
        n,
        n_rollers,
        l_rollers,
        fs,
        alpha,
        cxx=None,
        cyy=None,
        tag=None,
        n_link=None,
        scale_factor=1,
    ):

        Kb = 1.0e9
        kyy = (
            Kb
            * n_rollers**0.9
            * l_rollers**0.8
            * fs**0.1
            * (np.cos(alpha)) ** 1.9
        )

        nr = [8, 12, 16]
        ratio = [0.49, 0.66, 0.74]
        dict_ratio = dict(zip(nr, ratio))

        if n_rollers in dict_ratio.keys():
            kxx = dict_ratio[n_rollers] * kyy
        else:
            f = interpolate.interp1d(nr, ratio, "quadratic", fill_value="extrapolate")
            kxx = f(n_rollers) * kyy

        if cxx is None:
            cxx = 1.25e-5 * kxx
        if cyy is None:
            cyy = 1.25e-5 * kyy

        super().__init__(
            n=n,
            frequency=None,
            kxx=kxx,
            kxy=0.0,
            kyx=0.0,
            kyy=kyy,
            cxx=cxx,
            cxy=0.0,
            cyx=0.0,
            cyy=cyy,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
        )

        self.color = "#77ACA2"


class MagneticBearingElement(BearingElement):
    """Magnetic bearing.

    This class creates a magnetic bearing element.
    Converts electromagnetic parameters and PID gains to stiffness and damping
    coefficients.

    Parameters
    ----------
    n : int
        The node in which the magnetic bearing will be located in the rotor.
    g0 : float
        Air gap in m^2.
    i0 : float
        Bias current in Ampere
    ag : float
        Pole area in m^2.
    nw : float or int
        Number of windings
    alpha : float or int
        Pole angle in radians.
    kp_pid : float or int
        Proportional gain of the PID controller.
    kd_pid : float or int
        Derivative gain of the PID controller.
    k_amp : float or int
        Gain of the amplifier model.
    k_sense : float or int
        Gain of the sensor model.
    tag : str, optional
        A tag to name the element
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    ----------
    See the following reference for the electromagnetic parameters g0, i0, ag, nw, alpha:
    Book: Magnetic Bearings. Theory, Design, and Application to Rotating Machinery
    Authors: Gerhard Schweitzer and Eric H. Maslen
    Page: 69-80

    Examples
    --------
    >>> n = 0
    >>> g0 = 1e-3
    >>> i0 = 1.0
    >>> ag = 1e-4
    >>> nw = 200
    >>> alpha = 0.392
    >>> kp_pid = 1.0
    >>> kd_pid = 1.0
    >>> k_amp = 1.0
    >>> k_sense = 1.0
    >>> tag = "magneticbearing"
    >>> mbearing = MagneticBearingElement(n=n, g0=g0, i0=i0, ag=ag, nw=nw,alpha=alpha,
    ...                                   kp_pid=kp_pid, kd_pid=kd_pid, k_amp=k_amp,
    ...                                   k_sense=k_sense)
    >>> mbearing.kxx
    [-4640.623377181318]
    """

    def __init__(
        self,
        n,
        g0,
        i0,
        ag,
        nw,
        alpha,
        kp_pid,
        kd_pid,
        k_amp,
        k_sense,
        tag=None,
        n_link=None,
        scale_factor=1,
        **kwargs,
    ):
        self.g0 = g0
        self.i0 = i0
        self.ag = ag
        self.nw = nw
        self.alpha = alpha
        self.kp_pid = kp_pid
        self.kd_pid = kd_pid
        self.k_amp = k_amp
        self.k_sense = k_sense

        pL = [g0, i0, ag, nw, alpha, kp_pid, kd_pid, k_amp, k_sense]
        pA = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Check if it is a number or a list with 2 items
        for i in range(9):
            if type(pL[i]) == float or int:
                pA[i] = np.array(pL[i])
            else:
                if type(pL[i]) == list:
                    if len(pL[i]) > 2:
                        raise ValueError(
                            "Parameters must be scalar or a list with 2 items"
                        )
                    else:
                        pA[i] = np.array(pL[i])
                else:
                    raise ValueError("Parameters must be scalar or a list with 2 items")

        # From: "Magnetic Bearings. Theory, Design, and Application to Rotating Machinery"
        # Authors: Gerhard Schweitzer and Eric H. Maslen
        # Page: 343
        ks = (
            -4.0
            * pA[1] ** 2.0
            * np.cos(pA[4])
            * 4.0
            * np.pi
            * 1e-7
            * pA[3] ** 2.0
            * pA[2]
            / (4.0 * pA[0] ** 3)
        )
        ki = (
            4.0
            * pA[1]
            * np.cos(pA[4])
            * 4.0
            * np.pi
            * 1e-7
            * pA[3] ** 2.0
            * pA[2]
            / (4.0 * pA[0] ** 2)
        )
        k = ki * pA[7] * pA[8] * (pA[5] + np.divide(ks, ki * pA[7] * pA[8]))
        c = ki * pA[7] * pA[6] * pA[8]
        # k = ki * k_amp*k_sense*(kp_pid+ np.divide(ks, ki*k_amp*k_sense))
        # c = ki*k_amp*kd_pid*k_sense

        # Get the parameters from k and c
        if np.isscalar(k):
            # If k is scalar, symmetry is assumed
            kxx = k
            kyy = k
        else:
            kxx = k[0]
            kyy = k[1]

        if np.isscalar(c):
            # If c is scalar, symmetry is assumed
            cxx = c
            cyy = c
        else:
            cxx = c[0]
            cyy = c[1]

        super().__init__(
            n=n,
            frequency=None,
            kxx=kxx,
            kxy=0.0,
            kyx=0.0,
            kyy=kyy,
            cxx=cxx,
            cxy=0.0,
            cyx=0.0,
            cyy=cyy,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
        )


class CylindricalBearing(BearingElement):
    """Cylindrical hydrodynamic bearing.

    A cylindrical hydrodynamic bearing modeled as per
    :cite:`friswell2010dynamics` (page 177) assuming the following:

    - the flow is laminar and Reynolds’s equation applies
    - the bearing is very short, so that L /D << 1, where L is the bearing length and
    D is the bearing diameter, which means that the pressure gradients are much
    larger in the axial than in the circumferential direction
    - the lubricant pressure is zero at the edges of the bearing
    - the bearing is operating under steady running conditions
    - the lubricant properties do not vary substantially throughout the oil film
    - the shaft does not tilt in the bearing

    Parameters
    ----------
    n : int
        Node which the bearing will be located in.
    speed : list, pint.Quantity
        List with shaft speeds frequency (rad/s).
    weight : float, pint.Quantity
        Gravity load (N).
        It is a positive value in the -Y direction. For a symmetric rotor that is
        supported by two journal bearings, it is half of the total rotor weight.
    bearing_length : float, pint.Quantity
        Bearing axial length (m).
    journal_diameter : float, pint.Quantity
        Journal diameter (m).
    radial_clearance : float, pint.Quantity
        Bore assembly radial clearance (m).
    oil_viscosity : float, pint.Quantity
        Oil viscosity (Pa.s).

    Returns
    -------
        CylindricalBearing element.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> import ross as rs
    >>> Q_ = rs.Q_
    >>> cylindrical = CylindricalBearing(
    ...     n=0,
    ...     speed=Q_([1500, 2000], "RPM"),
    ...     weight=525,
    ...     bearing_length=Q_(30, "mm"),
    ...     journal_diameter=Q_(100, "mm"),
    ...     radial_clearance=Q_(0.1, "mm"),
    ...     oil_viscosity=0.1,
    ... )
    >>> cylindrical.K(Q_(1500, "RPM")) # doctest: +ELLIPSIS
    array([[ 12807959...,  16393593...],
           [-25060393...,   8815302...]])
    """

    @check_units
    def __init__(
        self,
        n,
        speed=None,
        weight=None,
        bearing_length=None,
        journal_diameter=None,
        radial_clearance=None,
        oil_viscosity=None,
        **kwargs,
    ):
        self.n = n

        self.speed = []
        for spd in speed:
            if spd == 0:
                # replace 0 speed with small value to avoid errors
                self.speed.append(0.1)
            else:
                self.speed.append(spd)
        self.weight = weight
        self.bearing_length = bearing_length
        self.journal_diameter = journal_diameter
        self.radial_clearance = radial_clearance
        self.oil_viscosity = oil_viscosity

        # modified Sommerfeld number or the Ocvirk number
        Ss = (
            journal_diameter
            * speed
            * oil_viscosity
            * bearing_length**3
            / (8 * radial_clearance**2 * weight)
        )

        self.modified_sommerfeld = Ss
        self.sommerfeld = (Ss / np.pi) * (journal_diameter / bearing_length) ** 2

        # find roots
        self.roots = []
        for s in Ss:
            poly = Polynomial(
                [
                    1,
                    -(4 + np.pi**2 * s**2),
                    (6 - s**2 * (16 - np.pi**2)),
                    -4,
                    1,
                ]
            )
            self.roots.append(poly.roots())

        # select real root between 0 and 1
        self.root = []
        for roots in self.roots:
            for root in roots:
                if (0 < root < 1) and np.isreal(root):
                    self.root.append(np.real(root))

        self.eccentricity = [np.sqrt(root) for root in self.root]
        self.attitude_angle = [
            np.arctan(np.pi * np.sqrt(1 - e**2) / 4 * e) for e in self.eccentricity
        ]

        coefficients = [
            "kxx",
            "kxy",
            "kyx",
            "kyy",
            "cxx",
            "cxy",
            "cyx",
            "cyy",
        ]
        coefficients_dict = {coeff: [] for coeff in coefficients}

        for e, spd in zip(self.eccentricity, self.speed):
            π = np.pi
            # fmt: off
            h0 = 1 / (π ** 2 * (1 - e ** 2) + 16 * e ** 2) ** (3 / 2)
            auu = h0 * 4 * (π ** 2 * (2 - e ** 2) + 16 * e ** 2)
            auv = h0 * π * (π ** 2 * (1 - e ** 2) ** 2 - 16 * e ** 4) / (e * np.sqrt(1 - e ** 2))
            avu = - h0 * π * (π ** 2 * (1 - e ** 2) * (1 + 2 * e ** 2) + 32 * e ** 2 * (1 + e ** 2)) / (e * np.sqrt(1 - e ** 2))
            avv = h0 * 4 * (π ** 2 * (1 + 2 * e ** 2) + 32 * e ** 2 * (1 + e ** 2) / (1 - e ** 2))
            buu = h0 * 2 * π * np.sqrt(1 - e ** 2) * (π ** 2 * (1 + 2 * e ** 2) - 16 * e ** 2) / e
            buv = bvu = -h0 * 8 * (π ** 2 * (1 + 2 * e ** 2) - 16 * e ** 2)
            bvv = h0 * 2 * π * (π ** 2 * (1 - e ** 2) ** 2 + 48 * e ** 2) / (e * np.sqrt(1 - e ** 2))
            # fmt: on
            for coeff, term in zip(
                coefficients, [auu, auv, avu, avv, buu, buv, bvu, bvv]
            ):
                if coeff[0] == "k":
                    coefficients_dict[coeff].append(weight / radial_clearance * term)
                elif coeff[0] == "c":
                    coefficients_dict[coeff].append(
                        weight / (radial_clearance * spd) * term
                    )

        super().__init__(self.n, frequency=self.speed, **coefficients_dict, **kwargs)


class BearingElement6DoF(BearingElement):
    """A generalistic 6 DoF bearing element.

    This class will create a bearing
    element based on the user supplied stiffness and damping coefficients. These
    are determined alternatively, via purposefully built codes.

    Parameters
    ----------
    kxx : float, array, pint.Quantity
        Direct stiffness in the x direction (N/m).
    cxx : float, array, pint.Quantity
        Direct damping in the x direction (N*s/m).
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        Defaults to kxx
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        Default is to cxx
    kxy : float, array, pint.Quantity, optional
        Cross stiffness between xy directions (N/m).
        Default is 0
    kyx : float, array, pint.Quantity, optional
        Cross stiffness between yx directions (N/m).
        Default is 0
    kzz : float, array, pint.Quantity, optional
        Direct stiffness in the z direction (N/m).
        Default is 0
    cxy : float, array, pint.Quantity, optional
        Cross damping between xy directions (N*s/m).
        Default is 0
    cyx : float, array, pint.Quantity, optional
        Cross damping between yx directions (N*s/m).
        Default is 0
    czz : float, array, pint.Quantity, optional
        Direct damping in the z direction (N*s/m).
        Default is 0
    frequency : array, pint.Quantity, optional
        Array with the frequencies (rad/s).
    tag : str, optional
        A tag to name the element
        Default is None
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.

    Examples
    --------
    >>> n = 0
    >>> kxx = 1.0e7
    >>> kyy = 1.5e7
    >>> kzz = 5.0e5
    >>> bearing = BearingElement6DoF(n=n, kxx=kxx, kyy=kyy, kzz=kzz,
    ...                              cxx=0, cyy=0)
    >>> bearing.K(0)
    array([[10000000.,        0.,        0.],
           [       0., 15000000.,        0.],
           [       0.,        0.,   500000.]])
    """

    @check_units
    def __init__(
        self,
        n,
        kxx,
        cxx,
        kyy=None,
        cyy=None,
        kxy=0.0,
        kyx=0.0,
        kzz=0.0,
        cxy=0.0,
        cyx=0.0,
        czz=0.0,
        frequency=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        color="#355d7a",
    ):
        super().__init__(
            n=n,
            kxx=kxx,
            cxx=cxx,
            kyy=kyy,
            kxy=kxy,
            kyx=kyx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
            frequency=frequency,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
            color=color,
        )

        new_args = ["kzz", "czz"]

        args_dict = locals()
        coefficients = {}

        if kzz is None:
            args_dict["kzz"] = kxx * 0.0
        if czz is None:
            args_dict["czz"] = cxx * 0.0

        # check coefficients len for consistency
        coefficients_len = []

        for arg in new_args:
            coefficient, interpolated = self._process_coefficient(args_dict[arg])
            setattr(self, arg, coefficient)
            setattr(self, f"{arg}_interpolated", interpolated)
            coefficients_len.append(len(coefficient))

        coefficients_len = [len(v.coefficient) for v in coefficients.values()]

        if frequency is not None and type(frequency) != float:
            coefficients_len.append(len(args_dict["frequency"]))
            if len(set(coefficients_len)) > 1:
                raise ValueError(
                    "Arguments (coefficients and frequency)"
                    " must have the same dimension"
                )
        else:
            for c in coefficients_len:
                if c != 1:
                    raise ValueError(
                        "Arguments (coefficients and frequency)"
                        " must have the same dimension"
                    )

    def __hash__(self):
        return hash(self.tag)

    def __repr__(self):
        """Return a string representation of a bearing element.

        Returns
        -------
        A string representation of a bearing element object.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing # doctest: +ELLIPSIS
        BearingElement(n=0, n_link=None,
         kxx=[...
        """
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n}, n_link={self.n_link},\n"
            f" kxx={self.kxx}, kxy={self.kxy},\n"
            f" kyx={self.kyx}, kyy={self.kyy},\n"
            f" kzz={self.kzz}, cxx={self.cxx},\n"
            f" cxy={self.cxy}, cyx={self.cyx},\n"
            f" cyy={self.cyy}, czz={self.czz},\n"
            f" frequency={self.frequency}, tag={self.tag!r})"
        )

    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other : object
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
            "kzz",
            "cxx",
            "cyy",
            "cxy",
            "cyx",
            "czz",
            "frequency",
        ]
        if isinstance(other, self.__class__):
            init_args = []
            for arg in signature(self.__init__).parameters:
                if arg not in ["kwargs"]:
                    init_args.append(arg)

            init_args_comparison = []
            for arg in init_args:
                comparison = getattr(self, arg) == getattr(other, arg)
                try:
                    comparison = all(comparison)
                except TypeError:
                    pass

                init_args_comparison.append(comparison)

            init_args_comparison = all(init_args_comparison)
            attributes_comparison = all(
                (
                    (
                        np.array(getattr(self, attr)) == np.array(getattr(other, attr))
                    ).all()
                    for attr in compared_attributes
                )
            )

            return init_args_comparison and attributes_comparison
        return False

    def save(self, file):
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        # remove some info before saving
        brg_data = self.__dict__.copy()
        params_to_remove = [
            "kxx_interpolated",
            "kyy_interpolated",
            "kxy_interpolated",
            "kyx_interpolated",
            "kzz_interpolated",
            "cxx_interpolated",
            "cyy_interpolated",
            "cxy_interpolated",
            "cyx_interpolated",
            "czz_interpolated",
            "n_l",
            "n_r",
            "dof_global_index",
        ]
        for p in params_to_remove:
            brg_data.pop(p)

        # change np.array to lists so that we can save in .toml as list(floats)
        params = [
            "kxx",
            "kyy",
            "kxy",
            "kyx",
            "kzz",
            "cxx",
            "cyy",
            "cxy",
            "cyx",
            "czz",
            "frequency",
        ]
        for p in params:
            try:
                brg_data[p] = [float(i) for i in brg_data[p]]
            except TypeError:
                pass

        data[f"{self.__class__.__name__}_{self.tag}"] = brg_data

        with open(file, "w") as f:
            toml.dump(data, f)

    @classmethod
    def load(cls, file):
        data = toml.load(file)
        # extract single dictionary in the data
        data = list(data.values())[0]
        params = [
            "kxx",
            "kyy",
            "kxy",
            "kyx",
            "kzz",
            "cxx",
            "cyy",
            "cxy",
            "cyx",
            "czz",
            "frequency",
            "n",
            "tag",
            "n_link",
            "scale_factor",
        ]
        kwargs = {}
        for p in params:
            try:
                kwargs[p] = data.pop(p)
            except KeyError:
                pass

        bearing = cls(**kwargs)
        for k, v in data.items():
            setattr(bearing, k, v)

        return bearing

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

        >>> bearing = bearing_6dof_example()
        >>> bearing.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2}
        """
        return dict(x_0=0, y_0=1, z_0=2)

    def K(self, frequency):
        """Stiffness matrix for an instance of a bearing element.

        This method returns the stiffness matrix for an instance of a bearing element.

        Parameters
        ----------
        frequency : float
            The excitation frequency (rad/s).

        Returns
        -------
        K : np.ndarray
            A 3x3 matrix of floats containing the kxx, kxy, kyx, kyy and kzz values (N/m).

        Examples
        --------
        >>> bearing = bearing_6dof_example()
        >>> bearing.K(0)
        array([[1000000.,       0.,       0.],
               [      0.,  800000.,       0.],
               [      0.,       0.,  100000.]])
        """
        kxx = self.kxx_interpolated(frequency)
        kyy = self.kyy_interpolated(frequency)
        kxy = self.kxy_interpolated(frequency)
        kyx = self.kyx_interpolated(frequency)
        kzz = self.kzz_interpolated(frequency)

        K = np.array([[kxx, kxy, 0], [kyx, kyy, 0], [0, 0, kzz]])

        return K

    def C(self, frequency):
        """Damping matrix for an instance of a bearing element.

        This method returns the damping matrix for an instance of a bearing element.

        Parameters
        ----------
        frequency : float
            The excitation frequency (rad/s).

        Returns
        -------
        C: np.ndarray
            A 3x3 matrix of floats containing the cxx, cxy, cyx, cyy, and czz values (N/m).

        Examples
        --------
        >>> bearing = bearing_6dof_example()
        >>> bearing.C(0)
        array([[200.,   0.,   0.],
               [  0., 150.,   0.],
               [  0.,   0.,  50.]])
        """
        cxx = self.cxx_interpolated(frequency)
        cyy = self.cyy_interpolated(frequency)
        cxy = self.cxy_interpolated(frequency)
        cyx = self.cyx_interpolated(frequency)
        czz = self.czz_interpolated(frequency)

        C = np.array([[cxx, cxy, 0], [cyx, cyy, 0], [0, 0, czz]])

        return C


def bearing_example():
    """Create an example of bearing element.

    This function returns an instance of a simple seal. The purpose is to make
    available a simple model so that doctest can be written using it.

    Returns
    -------
    An instance of a bearing object.

    Examples
    --------
    >>> bearing = bearing_example()
    >>> bearing.frequency[0]
    0.0
    """
    w = np.linspace(0, 200, 11)
    bearing = BearingElement(n=0, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2, frequency=w)
    return bearing


def seal_example():
    """Create an example of seal element.

    This function returns an instance of a simple seal. The purpose is to make
    available a simple model so that doctest can be written using it.

    Returns
    -------
    seal : ross.SealElement
        An instance of a bearing object.

    Examples
    --------
    >>> seal = seal_example()
    >>> seal.frequency[0]
    0.0
    """
    w = np.linspace(0, 200, 11)
    seal = SealElement(n=0, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2, frequency=w)
    return seal


def bearing_6dof_example():
    """Create an example of bearing element.

    This function returns an instance of a simple bearing. The purpose is to make
    available a simple model so that doctest can be written using it.

    Returns
    -------
    bearing : ross.BearingElement6DoF
        An instance of a bearing object.

    Examples
    --------
    >>> bearing = bearing_6dof_example()
    >>> bearing.kxx
    [1000000.0]"""
    bearing = BearingElement6DoF(
        n=0, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2, kzz=1e5, czz=0.5e2
    )
    return bearing
