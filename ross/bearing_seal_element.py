"""Bearing Element module.

This module defines the BearingElement classes which will be used to represent the rotor
bearings and seals. There are 7 different classes to represent bearings options.
"""

import control as ct
import numpy as np
import toml
import warnings
from inspect import signature
from prettytable import PrettyTable
from numpy.polynomial import Polynomial
from plotly import graph_objects as go
from scipy import interpolate as interpolate

from ross.element import Element
from ross.bearings import fluid_flow as flow
from ross.bearings.fluid_flow_coefficients import (
    calculate_stiffness_and_damping_coefficients,
)
from ross.units import Q_, check_units
from ross.utils import (
    read_table_file,
    is_scalar,
    is_scalar_or_list,
    is_transfer_function_or_none,
    is_list_or_none,
)

__all__ = [
    "BearingElement",
    "SealElement",
    "BallBearingElement",
    "RollerBearingElement",
    "BearingFluidFlow",
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
    mxx : float, array, pint.Quantity
        Direct mass in the x direction (kg).
        Default is 0.
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        Default is kxx.
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        Default is cxx.
    myy : float, array, pint.Quantity, optional
        Direct mass in the y direction (kg).
        Default is mxx.
    kxy : float, array, pint.Quantity, optional
        Cross coupled stiffness in the x direction (N/m).
        Default is 0.
    cxy : float, array, pint.Quantity, optional
        Cross coupled damping in the x direction (N*s/m).
        Default is 0.
    mxy : float, array, pint.Quantity, optional
        Cross coupled mass in the x direction (kg).
        Default is 0.
    kyx : float, array, pint.Quantity, optional
        Cross coupled stiffness in the y direction (N/m).
        Default is 0.
    cyx : float, array, pint.Quantity, optional
        Cross coupled damping in the y direction (N*s/m).
        Default is 0.
    myx : float, array, pint.Quantity, optional
        Cross coupled mass in the y direction (kg).
        Default is 0.
    kzz : float, array, pint.Quantity, optional
        Direct stiffness in the z direction (N/m).
        Default is 0.
    czz : float, array, pint.Quantity, optional
        Direct damping in the z direction (N*s/m).
        Default is 0.
    mzz : float, array, pint.Quantity, optional
        Direct mass in the z direction (kg).
        Default is 0.
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
    >>> bearing0.K(frequency[-1])
    array([[1000000.,       0.,       0.],
           [      0.,  800000.,       0.],
           [      0.,       0.,       0.]])
    >>> bearing0.C(frequency[-1])
    array([[200.,   0.,   0.],
           [  0., 150.,   0.],
           [  0.,   0.,   0.]])
    """

    @check_units
    def __init__(
        self,
        n,
        kxx,
        cxx,
        mxx=0,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        myy=None,
        mxy=0,
        myx=0,
        kzz=0,
        czz=0,
        mzz=0,
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

        if kyy is None:
            kyy = kxx
        if cyy is None:
            cyy = cxx
        if myy is None:
            myy = mxx

        args = [
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
            "mxx",
            "myy",
            "mxy",
            "myx",
            "mzz",
        ]

        # all args to coefficients.  output of locals() should be READ ONLY
        args_dict = locals()

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

    def _get_coefficient_list(self, ignore_mass=False):
        """List with all bearing coefficients as strings"""
        coefficients = [
            attr.replace("_interpolated", "")
            for attr in self.__dict__.keys()
            if "_interpolated" in attr
        ]

        if ignore_mass:
            coefficients = [coeff for coeff in coefficients if "m" not in coeff]

        return coefficients

    def plot(
        self,
        coefficients=None,
        frequency_units="rad/s",
        stiffness_units="N/m",
        damping_units="N*s/m",
        mass_units="kg",
        fig=None,
        **kwargs,
    ):
        """Plot coefficient vs frequency.

        Parameters
        ----------
        coefficients : list, str
            List or str with the coefficients to plot.
        frequency_units : str, optional
            Frequency units.
            Default is rad/s.
        stiffness_units : str, optional
            Stiffness units.
            Default is N/m.
        damping_units : str, optional
            Damping units.
            Default is N*s/m.
        mass_units : str, optional
            Mass units.
            Default is kg.
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
            raise ValueError(
                "Can only plot stiffness, damping or mass in the same plot."
            )

        coeff_to_plot = coefficients_set.pop()

        if coeff_to_plot == "k":
            default_units = "N/m"
            y_units = stiffness_units
        elif coeff_to_plot == "c":
            default_units = "N*s/m"
            y_units = damping_units
        else:
            default_units = "kg"
            y_units = mass_units

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

    @check_units
    def format_table(
        self,
        frequency=None,
        coefficients=None,
        frequency_units="rad/s",
        stiffness_units="N/m",
        damping_units="N*s/m",
        mass_units="kg",
    ):
        """Return frequency vs coefficients in table format.

        Parameters
        ----------
        frequency : array, pint.Quantity, optional
            Array with frequencies (rad/s).
            Default is 5 values from min to max frequency.
        coefficients : list, str, optional
            List or str with the coefficients to include.
            Defaults is a list of stiffness and damping coefficients.
        frequency_units : str, optional
            Frequency units.
            Default is rad/s.
        stiffness_units : str, optional
            Stiffness units.
            Default is N/m.
        damping_units : str, optional
            Damping units.
            Default is N*s/m.
        mass_units : str, optional
            Mass units.
            Default is kg.

        Returns
        -------
        table : PrettyTable object
            Table object with bearing coefficients to be printed.

        Example
        -------
        >>> bearing = bearing_example()
        >>> table = bearing.format_table(
        ...     frequency=[0, 50, 100, 150, 200],
        ...     coefficients=['kxx', 'kxy', 'cxx', 'cxy']
        ... )
        >>> print(table)
        +-------------------+-----------+-----------+-------------+-------------+
        | Frequency [rad/s] | kxx [N/m] | kxy [N/m] | cxx [N*s/m] | cxy [N*s/m] |
        +-------------------+-----------+-----------+-------------+-------------+
        |        0.0        | 1000000.0 |    0.0    |    200.0    |     0.0     |
        |        50.0       | 1000000.0 |    0.0    |    200.0    |     0.0     |
        |       100.0       | 1000000.0 |    0.0    |    200.0    |     0.0     |
        |       150.0       | 1000000.0 |    0.0    |    200.0    |     0.0     |
        |       200.0       | 1000000.0 |    0.0    |    200.0    |     0.0     |
        +-------------------+-----------+-----------+-------------+-------------+
        """
        if isinstance(coefficients, str):
            coefficients = [coefficients]
        elif coefficients is None:
            coefficients = self._get_coefficient_list(ignore_mass=True)

        default_units = {"k": "N/m", "c": "N*s/m", "m": "kg"}
        y_units = {"k": stiffness_units, "c": damping_units, "m": mass_units}

        if frequency is None:
            frequency = np.linspace(min(self.frequency), max(self.frequency), 5)
        frequency_range = Q_(frequency, "rad/s").to(frequency_units).m

        headers = [f"Frequency [{frequency_units}]"]
        data = [frequency_range]

        table = PrettyTable()

        for coeff in coefficients:
            headers.append(f"{coeff} [{default_units[coeff[0]]}]")
            columns = (
                Q_(
                    getattr(self, f"{coeff}_interpolated")(frequency),
                    default_units[coeff[0]],
                )
                .to(y_units[coeff[0]])
                .m
            )
            data.append(columns)

        table.field_names = headers
        for row in np.array(data).T:
            table.add_row(row.round(5))

        return table

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
            f" mxx={self.mxx}, mxy={self.mxy},\n"
            f" myx={self.myx}, myy={self.myy},\n"
            f" mzz={self.mzz},\n"
            f" frequency={self.frequency}, tag={self.tag!r})"
        )

    def __eq__(self, other):
        """Equality method for comparisons.

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
        attributes_comparison = False

        if isinstance(other, self.__class__):
            init_args = set(signature(self.__init__).parameters).intersection(
                self.__dict__.keys()
            )

            coefficients = set(self._get_coefficient_list())

            compared_attributes = list(coefficients.union(init_args))
            compared_attributes.sort()

            for attr in compared_attributes:
                self_attr = np.array(getattr(self, attr))
                other_attr = np.array(getattr(other, attr))

                if self_attr.shape == other_attr.shape:
                    attributes_comparison = (self_attr == other_attr).all()
                else:
                    attributes_comparison = False

                if not attributes_comparison:
                    return attributes_comparison

        return attributes_comparison

    def __hash__(self):
        return hash(self.tag)

    def save(self, file):
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        # save initialization args and coefficients
        init_args = set(signature(self.__init__).parameters).intersection(
            self.__dict__.keys()
        )

        coefficients = set(self._get_coefficient_list())

        args = list(coefficients.union(init_args))
        args.sort()

        brg_data = {arg: self.__dict__[arg] for arg in args}

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

        diff_args = set(signature(self.__init__).parameters).difference(
            self.__dict__.keys()
        )
        diff_args.discard("kwargs")

        class_name = (
            self.__class__.__name__
            if not diff_args
            else self.__class__.__bases__[0].__name__
        )

        data[f"{class_name}_{self.tag}"] = brg_data

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
        z_0 - axial translation

        >>> bearing = bearing_example()
        >>> bearing.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2}
        """
        return dict(x_0=0, y_0=1, z_0=2)

    def M(self, frequency):
        """Mass matrix for an instance of a bearing element.

        This method returns the mass matrix for an instance of a bearing
        element.

        Parameters
        ----------
        frequency : float
            The excitation frequency (rad/s).

        Returns
        -------
        M : np.ndarray
            Mass matrix (kg).

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.M(0)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        mxx = self.mxx_interpolated(frequency)
        myy = self.myy_interpolated(frequency)
        mxy = self.mxy_interpolated(frequency)
        myx = self.myx_interpolated(frequency)
        mzz = self.mzz_interpolated(frequency)

        M = np.array([[mxx, mxy, 0], [myx, myy, 0], [0, 0, mzz]])

        if self.n_link is not None:
            # fmt: off
            M = np.vstack((np.hstack([M, -M]),
                           np.hstack([-M, M])))
            # fmt: on

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
            A 3x3 matrix of floats containing the kxx, kxy, kyx, kyy and kzz
            values (N/m).

        Examples
        --------
        >>> bearing = bearing_example()
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
            A 3x3 matrix of floats containing the cxx, cxy, cyx, cyy, and czz
            values (N*s/m).

        Examples
        --------
        >>> bearing = bearing_example()
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
            A 3x3 matrix of floats.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> bearing.G()
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        G = np.zeros_like(self.K(0))

        return G

    def _hover_info(self, frequency=None):
        """Generate hover information for bearing element.

        This method can be overridden by subclasses to customize the hover
        information displayed when hovering over the bearing element in plots.

        Parameters
        ----------
        frequency : float, optional
            Frequency at which to display coefficients (rad/s).
            Not used - displays coefficients at first and last frequencies.

        Returns
        -------
        customdata : list
            Data to attach to hover trace.
        hovertemplate : str
            Template string for hover display with HTML formatting.

        Examples
        --------
        >>> bearing = bearing_example()
        >>> customdata, hovertemplate = bearing._hover_info()
        >>> customdata[0]  # node number
        0
        """
        # Get first and last frequencies
        if self.frequency is not None:
            if hasattr(self.frequency, "__iter__"):
                freq_0 = self.frequency[0]
                freq_1 = self.frequency[-1]
            else:
                freq_0 = freq_1 = self.frequency
        else:
            freq_0 = freq_1 = 0

        # Convert frequencies to RPM
        freq_0_rpm = Q_(freq_0, "rad/s").to("RPM").m
        freq_1_rpm = Q_(freq_1, "rad/s").to("RPM").m

        # Build hover template directly without intermediate list
        hovertemplate = f"Bearing at Node: {self.n}<br>"
        if self.tag is not None:
            hovertemplate = f"Tag: {self.tag}<br>" + hovertemplate

        hovertemplate += f"Frequency: {freq_0_rpm:.2f} ... {freq_1_rpm:.2f} RPM<br>"
        hovertemplate += (
            f"Kxx: {self.kxx_interpolated(freq_0):.3e} ... {self.kxx_interpolated(freq_1):.3e} N/m<br>"
            f"Kyy: {self.kyy_interpolated(freq_0):.3e} ... {self.kyy_interpolated(freq_1):.3e} N/m<br>"
            f"Cxx: {self.cxx_interpolated(freq_0):.3e} ... {self.cxx_interpolated(freq_1):.3e} N·s/m<br>"
            f"Cyy: {self.cyy_interpolated(freq_0):.3e} ... {self.cyy_interpolated(freq_1):.3e} N·s/m<br>"
        )

        # customdata is still needed for plotly, but can be minimal
        customdata = [self.n]

        return customdata, hovertemplate

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
        zpos, ypos, ypos_s, yc_pos = position

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
        fig.add_trace(go.Scatter(x=x_bot, y=np.add(yl_bot, yc_pos), **default_values))
        fig.add_trace(go.Scatter(x=x_bot, y=np.add(yu_bot, yc_pos), **default_values))

        # Add hover information marker at the center of bottom base
        customdata, hovertemplate = self._hover_info()
        # Scale marker size proportionally to the bearing icon height
        # icon_h already includes the scale_factor effect from rotor assembly
        marker_size = icon_h * 200  # proportional to actual bearing size
        hover_marker_values_top = dict(
            mode="markers",
            x=[zpos],
            y=[ypos + icon_h / 2],
            marker=dict(size=marker_size, color=self.color, opacity=0),
            customdata=[customdata],
            hovertemplate=hovertemplate,
            hoverinfo="text",
            name=self.tag,
            legendgroup="bearings",
            showlegend=False,
        )
        fig.add_trace(go.Scatter(**hover_marker_values_top))
        # copy the customdata and hovertemplate from the top marker just multiplying the y value by -1
        hover_marker_values_bottom = hover_marker_values_top.copy()
        hover_marker_values_bottom["y"] = [-1 * hover_marker_values_top["y"][0]]
        fig.add_trace(go.Scatter(**hover_marker_values_bottom))

        # plot top base
        x_top = [zpos, zpos, zs0, zs1]
        yl_top = [
            ypos + icon_h,
            ypos + 0.75 * icon_h,
            ypos + 0.75 * icon_h,
            ypos + 0.75 * icon_h,
        ]
        yu_top = [-y for y in yl_top]
        fig.add_trace(go.Scatter(x=x_top, y=np.add(yl_top, yc_pos), **default_values))
        fig.add_trace(go.Scatter(x=x_top, y=np.add(yu_top, yc_pos), **default_values))

        # plot ground
        if self.n_link is None:
            zl_g = [zs0 - step, zs1 + step]
            yl_g = [yl_top[0], yl_top[0]]
            yu_g = [-y for y in yl_g]
            fig.add_trace(go.Scatter(x=zl_g, y=np.add(yl_g, yc_pos), **default_values))
            fig.add_trace(go.Scatter(x=zl_g, y=np.add(yu_g, yc_pos), **default_values))

            step2 = (zl_g[1] - zl_g[0]) / n
            for i in range(n + 1):
                zl_g2 = [(zs0 - step) + step2 * (i), (zs0 - step) + step2 * (i + 1)]
                yl_g2 = [yl_g[0], 1.1 * yl_g[0]]
                yu_g2 = [-y for y in yl_g2]
                fig.add_trace(
                    go.Scatter(x=zl_g2, y=np.add(yl_g2, yc_pos), **default_values)
                )
                fig.add_trace(
                    go.Scatter(x=zl_g2, y=np.add(yu_g2, yc_pos), **default_values)
                )

        # plot spring
        z_spring = np.array([zs0, zs0, zs0, zs0])
        yl_spring = np.array([ys0, ys0 + step, ys0 + icon_w - step, ys0 + icon_w])

        for i in range(coils):
            z_spring = np.insert(z_spring, i + 2, zs0 - (-1) ** i * step)
            yl_spring = np.insert(yl_spring, i + 2, ys0 + (i + 1) * step)
        yu_spring = [-y for y in yl_spring]

        fig.add_trace(
            go.Scatter(x=z_spring, y=np.add(yl_spring, yc_pos), **default_values)
        )
        fig.add_trace(
            go.Scatter(x=z_spring, y=np.add(yu_spring, yc_pos), **default_values)
        )

        # plot damper - base
        z_damper1 = [zs1, zs1]
        yl_damper1 = [ys0, ys0 + 2 * step]
        yu_damper1 = [-y for y in yl_damper1]
        fig.add_trace(
            go.Scatter(x=z_damper1, y=np.add(yl_damper1, yc_pos), **default_values)
        )
        fig.add_trace(
            go.Scatter(x=z_damper1, y=np.add(yu_damper1, yc_pos), **default_values)
        )

        # plot damper - center
        z_damper2 = [zs1 - 2 * step, zs1 - 2 * step, zs1 + 2 * step, zs1 + 2 * step]
        yl_damper2 = [ys0 + 5 * step, ys0 + 2 * step, ys0 + 2 * step, ys0 + 5 * step]
        yu_damper2 = [-y for y in yl_damper2]
        fig.add_trace(
            go.Scatter(x=z_damper2, y=np.add(yl_damper2, yc_pos), **default_values)
        )
        fig.add_trace(
            go.Scatter(x=z_damper2, y=np.add(yu_damper2, yc_pos), **default_values)
        )

        # plot damper - top
        z_damper3 = [z_damper2[0], z_damper2[2], zs1, zs1]
        yl_damper3 = [
            ys0 + 4 * step,
            ys0 + 4 * step,
            ys0 + 4 * step,
            ypos + 1.5 * icon_w,
        ]
        yu_damper3 = [-y for y in yl_damper3]

        fig.add_trace(
            go.Scatter(x=z_damper3, y=np.add(yl_damper3, yc_pos), **default_values)
        )
        fig.add_trace(
            go.Scatter(x=z_damper3, y=np.add(yu_damper3, yc_pos), **default_values)
        )

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

    .. deprecated:: 2.0.0
        `BearingFluidFlow` is deprecated and will be removed in a future version.
        Use `PlainJournal` for advanced thermo-hydro-dynamic analysis with thermal
        effects, or `CylindricalBearing` for fast analytical calculations.

    This method always creates elements with frequency-dependent coefficients.
    It calculates a set of coefficients for each frequency value appendend to
    "omega".

    **Recommended alternatives:**

    - For advanced analysis with thermal effects, multi-pad configurations, and
      turbulence models, use :class:`PlainJournal` instead.
    - For quick calculations and preliminary design, use :class:`CylindricalBearing`
      instead.

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
    color : str, optional
        A color to be used when the element is represented.
        Default is '#355d7a'.

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
     kxx=[14...
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
        warnings.warn(
            "BearingFluidFlow is deprecated and will be removed in a future version. "
            "Use PlainJournal for advanced thermo-hydro-dynamic analysis with thermal effects, "
            "or CylindricalBearing for fast analytical calculations.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.nz = nz
        self.ntheta = ntheta
        self.length = length
        self.omega = omega
        self.p_in = p_in
        self.p_out = p_out
        self.radius_rotor = radius_rotor
        self.radius_stator = radius_stator
        self.visc = visc
        self.rho = rho
        self.eccentricity = eccentricity
        self.load = load

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
    mxx : float, array, pint.Quantity
        Direct mass in the x direction (kg).
        Default is 0.
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        Default is kxx.
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        Default is cxx.
    myy : float, array, pint.Quantity, optional
        Direct mass in the y direction (kg).
        Default is mxx.
    kxy : float, array, pint.Quantity, optional
        Cross coupled stiffness in the x direction (N/m).
        Default is 0.
    cxy : float, array, pint.Quantity, optional
        Cross coupled damping in the x direction (N*s/m).
        Default is 0.
    mxy : float, array, pint.Quantity, optional
        Cross coupled mass in the x direction (kg).
        Default is 0.
    kyx : float, array, pint.Quantity, optional
        Cross coupled stiffness in the y direction (N/m).
        Default is 0.
    cyx : float, array, pint.Quantity, optional
        Cross coupled damping in the y direction (N*s/m).
        Default is 0.
    myx : float, array, pint.Quantity, optional
        Cross coupled mass in the y direction (kg).
        Default is 0.
    kzz : float, array, pint.Quantity, optional
        Direct stiffness in the z direction (N/m).
        Default is 0.
    czz : float, array, pint.Quantity, optional
        Direct damping in the z direction (N*s/m).
        Default is 0.
    mzz : float, array, pint.Quantity, optional
        Direct mass in the z direction (kg).
        Default is 0.
    seal_leakage : float, optional
        Seal leakage.
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
        Default is 0.5.
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
    >>> seal.K(frequency[-1])
    array([[1000000.,       0.,       0.],
           [      0.,  800000.,       0.],
           [      0.,       0.,       0.]])
    >>> seal.C(frequency[-1])
    array([[200.,   0.,   0.],
           [  0., 150.,   0.],
           [  0.,   0.,   0.]])
    """

    @check_units
    def __init__(
        self,
        n,
        kxx,
        cxx,
        mxx=0,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        myy=None,
        mxy=0,
        myx=0,
        kzz=0,
        czz=0,
        mzz=0,
        frequency=None,
        seal_leakage=None,
        tag=None,
        n_link=None,
        scale_factor=None,
        color="#77ACA2",
        **kwargs,
    ):
        self.seal_leakage = seal_leakage

        super().__init__(
            n=n,
            frequency=frequency,
            kxx=kxx,
            kxy=kxy,
            kyx=kyx,
            kyy=kyy,
            kzz=kzz,
            cxx=cxx,
            cxy=cxy,
            cyx=cyx,
            cyy=cyy,
            czz=czz,
            mxx=mxx,
            mxy=mxy,
            myx=myx,
            myy=myy,
            mzz=mzz,
            tag=tag,
            n_link=n_link,
            color=color,
        )

        # make seals with half the bearing size as a default
        self.scale_factor = scale_factor if scale_factor else self.scale_factor / 2

    def _hover_info(self, frequency=None):
        """Generate hover information for seal element.

        Overrides the base class method to include seal-specific information
        such as cross-coupled coefficients and seal leakage.

        Parameters
        ----------
        frequency : float, optional
            Frequency at which to display coefficients (rad/s).
            Not used - displays coefficients at first and last frequencies.

        Returns
        -------
        customdata : list
            Data to attach to hover trace.
        hovertemplate : str
            Template string for hover display with HTML formatting.
        """
        # Get first and last frequencies
        if self.frequency is not None:
            if hasattr(self.frequency, "__iter__"):
                freq_0 = self.frequency[0]
                freq_1 = self.frequency[-1]
            else:
                freq_0 = freq_1 = self.frequency
        else:
            freq_0 = freq_1 = 0

        # Convert frequencies to RPM
        freq_0_rpm = Q_(freq_0, "rad/s").to("RPM").m
        freq_1_rpm = Q_(freq_1, "rad/s").to("RPM").m

        # Build hover template directly
        hovertemplate = f"Seal at Node: {self.n}<br>"
        if self.tag is not None:
            hovertemplate = f"Tag: {self.tag}<br>" + hovertemplate

        hovertemplate += f"Frequency: {freq_0_rpm:.2f} ... {freq_1_rpm:.2f} RPM<br>"
        hovertemplate += (
            f"Kxx: {self.kxx_interpolated(freq_0):.3e} ... {self.kxx_interpolated(freq_1):.3e} N/m<br>"
            f"Kyy: {self.kyy_interpolated(freq_0):.3e} ... {self.kyy_interpolated(freq_1):.3e} N/m<br>"
            f"Kxy: {self.kxy_interpolated(freq_0):.3e} ... {self.kxy_interpolated(freq_1):.3e} N/m<br>"
            f"Kyx: {self.kyx_interpolated(freq_0):.3e} ... {self.kyx_interpolated(freq_1):.3e} N/m<br>"
            f"Cxx: {self.cxx_interpolated(freq_0):.3e} ... {self.cxx_interpolated(freq_1):.3e} N·s/m<br>"
            f"Cyy: {self.cyy_interpolated(freq_0):.3e} ... {self.cyy_interpolated(freq_1):.3e} N·s/m<br>"
            f"Cxy: {self.cxy_interpolated(freq_0):.3e} ... {self.cxy_interpolated(freq_1):.3e} N·s/m<br>"
            f"Cyx: {self.cyx_interpolated(freq_0):.3e} ... {self.cyx_interpolated(freq_1):.3e} N·s/m<br>"
        )

        if self.seal_leakage is not None:
            hovertemplate += f"Seal Leakage: {self.seal_leakage:.3e}<br>"

        customdata = [self.n]

        return customdata, hovertemplate


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
    color : str, optional
        A color to be used when the element is represented.
        Default is '#355d7a'.

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
    array([[4.64168838e+07, 0.00000000e+00, 0.00000000e+00],
           [0.00000000e+00, 1.00906269e+08, 0.00000000e+00],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
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
        color="#355d7a",
        **kwargs,
    ):
        self.n_balls = n_balls
        self.d_balls = d_balls
        self.fs = fs
        self.alpha = alpha

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
            color=color,
        )

    def _hover_info(self, frequency=None):
        """Generate hover information for ball bearing element.

        Overrides the base class method to include ball bearing-specific
        geometric and operating parameters.

        Parameters
        ----------
        frequency : float, optional
            Frequency at which to display coefficients (rad/s).
            Not used for ball bearings (frequency-independent).

        Returns
        -------
        customdata : list
            Data to attach to hover trace.
        hovertemplate : str
            Template string for hover display with HTML formatting.
        """
        hovertemplate = f"Ball Bearing at Node: {self.n}<br>"
        if self.tag is not None:
            hovertemplate = f"Tag: {self.tag}<br>" + hovertemplate

        hovertemplate += (
            f"Number of Balls: {self.n_balls:.0f}<br>"
            f"Ball Diameter: {self.d_balls:.4f} m<br>"
            f"Static Load: {self.fs:.2f} N<br>"
            f"Contact Angle: {self.alpha:.3f} rad<br>"
            f"Kxx: {self.kxx[0]:.3e} N/m<br>"
            f"Kyy: {self.kyy[0]:.3e} N/m<br>"
            f"Cxx: {self.cxx[0]:.3e} N·s/m<br>"
            f"Cyy: {self.cyy[0]:.3e} N·s/m<br>"
        )

        customdata = [self.n]

        return customdata, hovertemplate


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
    color : str, optional
        A color to be used when the element is represented.
        Default is '#355d7a'.

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
    array([[2.72821927e+08, 0.00000000e+00, 0.00000000e+00],
           [0.00000000e+00, 5.56779444e+08, 0.00000000e+00],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
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
        color="#355d7a",
        **kwargs,
    ):
        self.n_rollers = n_rollers
        self.l_rollers = l_rollers
        self.fs = fs
        self.alpha = alpha

        Kb = 1.0e9
        kyy = Kb * n_rollers**0.9 * l_rollers**0.8 * fs**0.1 * (np.cos(alpha)) ** 1.9

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
            color=color,
        )


class MagneticBearingElement(BearingElement):
    """
    Magnetic Bearing Element.

    This class represents an active magnetic bearing (AMB) modeled from
    its electromagnetic and control parameters. It automatically converts
    the physical parameters of the electromagnet and the controller
    (PID or custom transfer function) into equivalent stiffness and damping
    coefficients as functions of frequency.

    If a transfer function is not supplied, the user must define the
    proportional, integral, and derivative gains of the PID controller,
    including the cutoff frequency of the derivative filter.
    The class also computes the transformation of the stiffness and damping
    matrices according to the magnetic pole angle.

    Parameters
    ----------
    n : int
        Node index where the magnetic bearing is mounted on the rotor.
    g0 : float
        Nominal air gap (m).
    i0 : float or list of float
        Bias current applied to the coils (A). Can be a scalar value
        or a list of two values for the x and y axes.
    ag : float
        Effective pole area (m²).
    nw : int or float
        Number of turns per coil (windings).
    frequency : array_like, optional
        Frequency vector in rad/s used to evaluate the controller
        frequency response and build the equivalent stiffness and damping
        coefficients. If not provided, it is set automatically as a
        logarithmic vector between 10⁰ and 10⁴ rad/s.
    alpha : float, optional
        Angular position of the magnetic pole relative to the rotor x
        axis (radians). Default is 0.39269908 (approximately 22.5°).
    k_amp : float or list of float, optional
        Power amplifier gain (V/A). Can be a scalar value or a list of
        two values, one for each axis. Default is 1.
    k_sense : float or list of float, optional
        Displacement sensor gain (V/m). Can be a scalar value or a list
        of two values, one for each axis. Default is 1.
    kp_pid : float or int, optional
        Proportional gain of the PID controller. Default is 0.
    kd_pid : float or int, optional
        Derivative gain of the PID controller. Default is 0.
    ki_pid : float or int, optional
        Integral gain of the PID controller. Default is 0.
    n_f : float, optional
        Cutoff frequency of the derivative low-pass filter (rad/s)
        used in the PID controller. Default is 10 000.
    controller_transfer_function : control.TransferFunction, optional
        Continuous-time transfer function that represents a custom
        controller associated with the AMB. When provided, it overrides
        the PID gains for the computation of the controller frequency
        response and the equivalent coefficients.
    sensors_axis_rotation : float, optional
        Angular rotation between the rotor x–y axes and the sensor/actuator
        axes (radians). This angle is used to transform the equivalent
        isotropic stiffness and damping into the anisotropic matrices
        in the rotor coordinates. Default is 0.78539816
        (approximately 45°).
    tag : str, optional
        Label used to identify the element in the rotor model.
    n_link : int, optional
        Node connected to the bearing. If not specified, the bearing
        is considered grounded. Default is None.
    scale_factor : float, optional
        Scale factor for graphical representation. Default is 1.
    color : str, optional
        Color used for graphical representation. Default is "#355d7a".
    **kwargs
        Additional keyword arguments forwarded to the base class
        constructor.

    Attributes
    ----------
    g0 : float
        Nominal air gap (m).
    i0 : float or numpy.ndarray
        Bias current applied to the coils (A) in each controlled axis.
    ag : float
        Effective pole area (m²).
    nw : float
        Number of turns per coil (windings).
    alpha : float
        Magnetic pole angle relative to the rotor x axis (radians).
    k_amp : numpy.ndarray
        Amplifier gains for each axis (V/A).
    k_sense : numpy.ndarray
        Sensor gains for each axis (V/m).
    kp_pid, kd_pid, ki_pid : float
        PID controller gains.
    n_f : float
        Cutoff frequency of the derivative filter (rad/s).
    sensors_axis_rotation : float
        Rotation angle between rotor and sensor/actuator axes (radians).
    frequency : numpy.ndarray
        Frequency vector in rad/s used to compute the equivalent
        stiffness and damping matrices.
    ks : float
        Negative electromagnetic stiffness constant obtained from the
        linearization of the magnetic force.
    ki : float
        Electromagnetic current-to-force gain.
    kxx, kxy, kyx, kyy : numpy.ndarray
        Frequency-dependent stiffness coefficients in the rotor x–y
        coordinates (N/m).
    cxx, cxy, cyx, cyy : numpy.ndarray
        Frequency-dependent damping coefficients in the rotor x–y
        coordinates (N·s/m).
    controller_transfer_function_num : numpy.ndarray or None
        Numerator coefficients of the custom controller transfer
        function, if provided.
    controller_transfer_function_den : numpy.ndarray or None
        Denominator coefficients of the custom controller transfer
        function, if provided.
    A_c, B_c, C_c, D_c : numpy.ndarray or None
        Discrete-time state-space matrices of the controller model
        obtained after discretization.
    x_c : list of numpy.matrix or None
        List with two state vectors (one for each AMB axis) used in
        the discrete-time controller update.
    control_signal : list
        Time history of the control current signals, stored as a list
        of pairs of lists, one pair for each time step.
    magnetic_force_xy : list
        Time history of the magnetic forces expressed in the rotor
        x–y coordinates.
    magnetic_force_vw : list
        Time history of the magnetic forces expressed in the local
        pole coordinates.

    Notes
    -----
    - The electromagnetic coefficients ks and ki are computed following
      Schweitzer and Maslen, Magnetic Bearings: Theory, Design, and
      Application to Rotating Machinery, Springer, pages 69–80 and 343.
    - The frequency-dependent stiffness and damping coefficients are
      obtained from the real and imaginary parts of the closed-loop
      controller frequency response.
    - The same controller is assumed for both controlled axes.
    - The method get_analog_controller builds the continuous-time
      controller associated with the element, build_controller creates
      the discrete-time state-space representation, and compute_pid_amb
      uses this representation to compute the magnetic forces for
      time-domain simulations.

    See Also
    --------
    get_analog_controller
        Build or retrieve the analog controller transfer function.
    build_controller
        Discretize the analog controller and initialize the internal
        state-space representation.
    compute_pid_amb
        Compute the control force generated by the AMB controller
        for a single axis.

    Examples
    --------
    Create an AMB element using PID gains and inspect one of the
    equivalent stiffness coefficients:

        >>> import ross as rs
        >>> n = 0
        >>> g0 = 1e-3
        >>> i0 = 1.0
        >>> ag = 1e-4
        >>> nw = 200
        >>> alpha = 0.39269908
        >>> kp_pid = 1.0
        >>> kd_pid = 1.0
        >>> k_amp = 1.0
        >>> k_sense = 1.0
        >>> mb = rs.MagneticBearingElement(
        ...     n=n, g0=g0, i0=i0, ag=ag, nw=nw, alpha=alpha,
        ...     kp_pid=kp_pid, kd_pid=kd_pid, k_amp=k_amp, k_sense=k_sense
        ... )
        >>> mb.kxx[0]  # doctest: +ELLIPSIS
        -4639.28...

    Use a custom controller transfer function instead of PID gains:

        >>> import control as ct
        >>> import numpy as np
        >>> C_s = ct.TransferFunction([1.0, 10.0], [1.0, 2.0, 3.0])
        >>> mb_custom = rs.MagneticBearingElement(
        ...     n=0, g0=1e-3, i0=1.0, ag=1e-4, nw=200,
        ...     controller_transfer_function=C_s
        ... )
        >>> isinstance(mb_custom.get_analog_controller(), ct.TransferFunction)
        True
    """

    s = ct.TransferFunction.s

    def __init__(
        self,
        n,
        g0,
        i0,
        ag,
        nw,
        frequency=None,
        alpha=0.39269908,
        k_amp=1,
        k_sense=1,
        kp_pid=0,
        kd_pid=0,
        ki_pid=0,
        n_f=10_000,
        controller_transfer_function=None,
        sensors_axis_rotation=0.78539816,
        tag=None,
        n_link=None,
        scale_factor=1,
        color="#355d7a",
        **kwargs,
    ):
        self.g0 = is_scalar(g0, "g0")
        self.i0 = is_scalar_or_list(i0, 2, "i0")
        self.ag = is_scalar(ag, "ag")
        self.nw = is_scalar(nw, "nw")
        self.alpha = is_scalar(alpha, "alpha")
        self.kp_pid = is_scalar(kp_pid, "kp_pid")
        self.kd_pid = is_scalar(kd_pid, "kd_pid")
        self.ki_pid = is_scalar(ki_pid, "ki_pid")
        self.n_f = is_scalar(n_f, "n_f")
        self.k_amp = is_scalar_or_list(k_amp, 2, "k_amp")
        self.k_sense = is_scalar_or_list(k_sense, 2, "k_sense")
        is_transfer_function_or_none(
            controller_transfer_function, "controller_transfer_function"
        )
        self.sensors_axis_rotation = is_scalar(
            sensors_axis_rotation, "sensors_axis_rotation"
        )
        is_list_or_none(frequency, "frequency")

        self.controller_transfer_function_num = (
            np.array(controller_transfer_function.num).squeeze()
            if controller_transfer_function is not None
            else None
        )
        self.controller_transfer_function_den = (
            np.array(controller_transfer_function.den).squeeze()
            if controller_transfer_function is not None
            else None
        )

        # Control system (state matrices and state vector)
        self.A_c = None
        self.B_c = None
        self.C_c = None
        self.D_c = None
        self.x_c = None

        if (
            self.kp_pid == 0
            and self.ki_pid == 0
            and self.kd_pid == 0
            and controller_transfer_function is None
        ):
            raise ValueError(
                "You need to provide either the gains k_p, k_i, and k_d, or the transfer function of the "
                "controller you intend to associate with the Magnetic Bearing. Neither has been provided."
            )

        # From: "Magnetic Bearings. Theory, Design, and Application to Rotating Machinery"
        # Authors: Gerhard Schweitzer and Eric H. Maslen
        # Page: 343
        ks = (
            -4.0
            * self.i0**2.0
            * np.cos(self.alpha)
            * 4.0
            * np.pi
            * 1e-7
            * self.nw**2.0
            * self.ag
            / (4.0 * self.g0**3)
        )

        ki = (
            4.0
            * self.i0
            * np.cos(self.alpha)
            * 4.0
            * np.pi
            * 1e-7
            * self.nw**2.0
            * self.ag
            / (4.0 * self.g0**2)
        )

        self.ks = ks
        self.ki = ki
        self.control_signal = []
        self.magnetic_force_xy = []
        self.magnetic_force_vw = []

        C_s = self.get_analog_controller()
        omega = frequency if frequency is not None else np.logspace(0, 4)
        mag, phase, _ = ct.frequency_response(C_s, omega)
        Hjw = (mag * np.exp(1j * phase)).squeeze()
        C_real = Hjw.real
        C_imag = Hjw.imag

        k_eq = ks + ki * self.k_amp * self.k_sense * C_real
        c_eq = ki * self.k_amp * self.k_sense * C_imag * np.divide(1, omega)

        rotation_matrix = np.matrix(
            [
                [
                    np.cos(self.sensors_axis_rotation),
                    np.sin(self.sensors_axis_rotation),
                ],
                [
                    -np.sin(self.sensors_axis_rotation),
                    np.cos(self.sensors_axis_rotation),
                ],
            ]
        )
        inv_rotation_matrix = rotation_matrix.I

        k_xx = []
        k_xy = []
        k_yx = []
        k_yy = []
        c_xx = []
        c_xy = []
        c_yx = []
        c_yy = []
        for omega_i, k, c in zip(omega, k_eq, c_eq):
            k_equivalent_matrix = np.matrix([[k, 0], [0, k]])
            c_equivalent_matrix = np.matrix([[c, 0], [0, c]])

            k_xy_axis_matrix = (
                inv_rotation_matrix * k_equivalent_matrix * rotation_matrix
            )
            c_xy_axis_matrix = (
                inv_rotation_matrix * c_equivalent_matrix * rotation_matrix
            )

            k_xx.append(k_xy_axis_matrix[0, 0])
            k_xy.append(k_xy_axis_matrix[0, 1])
            k_yx.append(k_xy_axis_matrix[1, 0])
            k_yy.append(k_xy_axis_matrix[1, 1])

            c_xx.append(c_xy_axis_matrix[0, 0])
            c_xy.append(c_xy_axis_matrix[0, 1])
            c_yx.append(c_xy_axis_matrix[1, 0])
            c_yy.append(c_xy_axis_matrix[1, 1])

        super().__init__(
            n=n,
            frequency=omega,
            kxx=k_xx,
            kxy=k_xy,
            kyx=k_yx,
            kyy=k_yy,
            cxx=c_xx,
            cxy=c_xy,
            cyx=c_yx,
            cyy=c_yy,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
            color=color,
        )

    def _hover_info(self, frequency=None):
        """Generate hover information for magnetic bearing element.

        Overrides the base class method to include magnetic bearing-specific
        electromagnetic and control parameters.

        Parameters
        ----------
        frequency : float, optional
            Frequency at which to display coefficients (rad/s).
            Not used for magnetic bearings (frequency-independent).

        Returns
        -------
        customdata : list
            Data to attach to hover trace.
        hovertemplate : str
            Template string for hover display with HTML formatting.
        """
        hovertemplate = f"Magnetic Bearing at Node: {self.n}<br>"
        if self.tag is not None:
            hovertemplate = f"Tag: {self.tag}<br>" + hovertemplate

        hovertemplate += (
            f"Air Gap (g0): {self.g0:.4e} m<br>"
            f"Bias Current (i0): {self.i0:.2f} A<br>"
            f"Pole Area: {self.ag:.4e} m²<br>"
            f"Windings: {self.nw:.0f}<br>"
            f"PID Kp: {self.kp_pid:.3e}<br>"
            f"PID Kd: {self.kd_pid:.3e}<br>"
            f"Kxx: {self.kxx[0]:.3e} N/m<br>"
            f"Kyy: {self.kyy[0]:.3e} N/m<br>"
            f"Cxx: {self.cxx[0]:.3e} N·s/m<br>"
            f"Cyy: {self.cyy[0]:.3e} N·s/m<br>"
        )

        customdata = [self.n]

        return customdata, hovertemplate

    def compute_pid_amb(self, current_offset, setpoint, disp, dof_index):
        """Compute AMB control force for one axis using the discrete controller.

        This routine evaluates the discrete-time controller output for the selected
        degree of freedom (x or y) and converts it to the corresponding magnetic
        force. The controller must have been previously discretized and initialized
        with build_controller so that the internal state-space matrices and
        state vectors are available.

        The calculation follows:
        1) Compute the control output for the current error
           u = C_c @ x_c[dof_index] + D_c * error, where
           error = setpoint - disp.
        2) Update the controller state
           x_c[dof_index] = A_c @ x_c[dof_index] + B_c * error.
        3) Form the coil signal
           signal_pid = current_offset + u.
        4) Map current to force
           F = ki * signal_pid + ks * disp.

        Parameters
        ----------
        current_offset : float
            Static offset added to the controller signal (for example, bias current).
        setpoint : float
            Reference position for the controlled axis (usually zero).
        disp : float
            Measured displacement at the controlled axis.
        dof_index : int
            Axis selector within the AMB element: 0 for x, 1 for y.

        Returns
        -------
        float
            Magnetic force for the selected AMB axis.

        Notes
        -----
        - build_controller(dt) must be called beforehand; this method expects
          the attributes A_c, B_c, C_c, D_c and the list
          x_c to be initialized.
        - The electromagnetic coefficients ki (current-to-force gain) and
          ks (negative stiffness) must be defined in the element.
        - This method appends the latest control signal to
          self.control_signal[-1][dof_index]. Ensure that
          self.control_signal has a trailing item shaped like [[], []]
          (one list per axis) before the first call in a new time step.

        Examples
        --------
        >>> import ross as rs
        >>> import numpy as np
        >>> mb = rs.MagneticBearingElement(
        ...     n=0, g0=1e-3, i0=1.0, ag=1e-4, nw=200,
        ...     kp_pid=1.0, ki_pid=5.0, kd_pid=0.01, n_f=1e4
        ... )
        >>> mb.build_controller(dt=1e-3)
        >>> # start a new logging bucket for this time step (x and y)
        >>> mb.control_signal.append([[], []])
        >>> # compute force for x-axis given a small measured displacement
        >>> force_x = mb.compute_pid_amb(
        ...     current_offset=0.0, setpoint=0.0, disp=2e-4, dof_index=0
        ... )
        >>> isinstance(force_x, float)
        True

        """
        err = setpoint - disp
        u = self.C_c * self.x_c[dof_index] + self.D_c * err
        self.x_c[dof_index] = self.A_c * self.x_c[dof_index] + self.B_c * err

        signal_pid = current_offset + u
        magnetic_force = self.ki * signal_pid + self.ks * disp

        self.control_signal[dof_index].append(signal_pid.item())
        return magnetic_force.item()

    def get_analog_controller(self):
        """
        Return the continuous-time (analog) controller transfer function.

        This method builds or retrieves the analog controller C(s) associated with
        the magnetic bearing element. If a custom controller transfer function has
        been provided, it will be returned directly. Otherwise, the method constructs
        a PID controller with a low-pass filter applied to the derivative term,
        expressed as:

            C(s) = Kp + Ki / s + Kd * s * (1 / (1 + (1 / n_f) * s))

        where:
            - Kp is the proportional gain
            - Ki is the integral gain
            - Kd is the derivative gain
            - n_f is the cutoff frequency of the derivative filter

        Returns
        -------
        control.TransferFunction
            Continuous-time controller transfer function C(s).

        Notes
        -----
        - If both a custom transfer function and PID gains are defined,
          the custom transfer function takes precedence.
        - The Laplace variable 's' is obtained from MagneticBearingElement.s.
        - The output of this method is used in build_controller() to generate
          the corresponding discrete-time (sampled) version of the controller.

        See Also
        --------
        build_controller : Discretizes the analog controller and initializes its state matrices.
        compute_pid_amb : Uses the controller to compute active magnetic bearing control forces.

        Examples
        --------
        >>> import control as ct
        >>> import ross as rs
        >>> mb = rs.MagneticBearingElement(
        ...     n=0, g0=1e-3, i0=1.0, ag=1e-4, nw=200,
        ...     kp_pid=1.0, ki_pid=10.0, kd_pid=0.01, n_f=1e4
        ... )
        >>> C_s = mb.get_analog_controller()
        >>> isinstance(C_s, ct.TransferFunction)
        True

        >>> # Example with custom transfer function
        >>> mb.controller_transfer_function_num = [1, 5]
        >>> mb.controller_transfer_function_den = [1, 2, 3]
        >>> C_custom = mb.get_analog_controller()
        >>> C_custom.num[0][0]
        array([1, 5])
        """
        if (
            self.controller_transfer_function_num is None
            or self.controller_transfer_function_den is None
        ):
            s = MagneticBearingElement.s
            C_s = (
                self.kp_pid
                + self.ki_pid / s
                + self.kd_pid * s * (1 / (1 + (1 / self.n_f) * s))
            )
        else:
            C_s = ct.TransferFunction(
                self.controller_transfer_function_num,
                self.controller_transfer_function_den,
            )

        return C_s

    def build_controller(self, dt):
        """
        Discretize the analog controller and initialize its state-space representation.

        This method retrieves the continuous-time (analog) controller from the
        get_analog_controller() method, discretizes it using the Tustin (bilinear)
        transformation for the specified sampling period, converts it to a
        state-space representation, and initializes the internal controller states
        for the two active magnetic bearing (AMB) axes (x and y).

        Parameters
        ----------
        dt : float
            Sampling period in seconds used to discretize the controller.

        Effects
        --------
        This method updates the following object attributes:

        - A_c : numpy.ndarray
          Discrete-time state matrix of the controller (shape: (n_x, n_x)).
        - B_c : numpy.ndarray
          Discrete-time input matrix (shape: (n_x, n_u)).
        - C_c : numpy.ndarray
          Discrete-time output matrix (shape: (n_y, n_x)).
        - D_c : numpy.ndarray
          Discrete-time feedthrough matrix (shape: (n_y, n_u)).
        - x_c : list of numpy.matrix
          List with two zero-initialized state vectors (one for each AMB axis), each
          of size (n_x, 1).

        Notes
        -----
        - The controller is discretized using control.sample_system() with the
          method set to "tustin".
        - The resulting discrete-time controller is represented in state-space form
          using control.ss().
        - This method must be called before executing any function that updates the
          controller states, such as compute_pid_amb().

        See Also
        --------
        get_analog_controller : Builds or returns the analog controller transfer function.
        compute_pid_amb : Computes the control force generated by the AMB controller.

        Examples
        --------
        >>> import ross as rs
        >>> mb = rs.MagneticBearingElement(
        ...     n=0, g0=1e-3, i0=1.0, ag=1e-4, nw=200,
        ...     kp_pid=1.0, kd_pid=0.1, ki_pid=0.5, n_f=1e4
        ... )
        >>> mb.build_controller(dt=1e-3)
        >>> mb.A_c.shape[0] == mb.x_c[0].shape[0]
        True
        """
        C_s = self.get_analog_controller()
        C_z = ct.sample_system(C_s, dt, method="tustin")
        C_z_ss = ct.ss(C_z)
        self.A_c = C_z_ss.A
        self.B_c = C_z_ss.B
        self.C_c = C_z_ss.C
        self.D_c = C_z_ss.D

        self.x_c = [
            np.matrix(np.zeros((self.A_c.shape[0], 1))) for _ in range(2)
        ]  # for x and y directions


class CylindricalBearing(BearingElement):
    """Cylindrical hydrodynamic bearing - Simplified analytical model.

    This class provides a **fast, simplified** bearing model suitable for preliminary
    design and basic analysis. Uses closed-form analytical solutions from
    :cite:`friswell2010dynamics` (page 177).

    **When to use this class:**
    - Quick calculations and preliminary design
    - Educational purposes and basic understanding
    - Simple bearing geometries
    - When computational speed is critical

    **For advanced analysis, consider using PlainJournal instead, which provides:**
    - Thermo-hydro-dynamic (THD) effects
    - Multiple bearing geometries (circular, lobe, elliptical)
    - Multi-pad configurations
    - Turbulence models
    - Oil starvation/flooding conditions

    Assumptions
    -----------
    - the flow is laminar and Reynolds's equation applies
    - the bearing is very short, so that L /D << 1, where L is the bearing length and
    D is the bearing diameter, which means that the pressure gradients are much
    larger in the axial than in the circumferential direction
    - the lubricant pressure is zero at the edges of the bearing
    - the bearing is operating under steady running conditions
    - the lubricant properties do not vary substantially throughout the oil film (isothermal)
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
    tag : str, optional
        A tag to name the element
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#355d7a'.

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
        tag=None,
        scale_factor=1,
        color="#355d7a",
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

        super().__init__(
            n,
            frequency=self.speed,
            tag=tag,
            scale_factor=scale_factor,
            color=color,
            **coefficients_dict,
            **kwargs,
        )

    def _hover_info(self, frequency=None):
        """Generate hover information for cylindrical bearing element.

        Overrides the base class method to include cylindrical bearing-specific
        fluid film parameters and operating conditions.

        Parameters
        ----------
        frequency : float, optional
            Frequency at which to display coefficients (rad/s).
            Not used - displays coefficients at first and last speeds.

        Returns
        -------
        customdata : list
            Data to attach to hover trace.
        hovertemplate : str
            Template string for hover display with HTML formatting.
        """
        # Get first and last speeds
        freq_0 = self.speed[0]
        freq_1 = self.speed[-1]
        idx_0 = 0
        idx_1 = -1

        # Convert speeds to RPM
        freq_0_rpm = Q_(freq_0, "rad/s").to("RPM").m
        freq_1_rpm = Q_(freq_1, "rad/s").to("RPM").m

        hovertemplate = f"Cylindrical Bearing at Node: {self.n}<br>"
        if self.tag is not None:
            hovertemplate = f"Tag: {self.tag}<br>" + hovertemplate

        hovertemplate += (
            f"Length: {self.bearing_length:.4f} m<br>"
            f"Journal Diameter: {self.journal_diameter:.4f} m<br>"
            f"Radial Clearance: {self.radial_clearance:.4e} m<br>"
            f"Oil Viscosity: {self.oil_viscosity:.4f} Pa·s<br>"
            f"Load: {self.weight:.2f} N<br>"
            f"Speed: {freq_0_rpm:.2f} ... {freq_1_rpm:.2f} RPM<br>"
            f"Eccentricity: {self.eccentricity[idx_0]:.4f} ... {self.eccentricity[idx_1]:.4f}<br>"
            f"Attitude Angle: {self.attitude_angle[idx_0]:.4f} ... {self.attitude_angle[idx_1]:.4f} rad<br>"
            f"Sommerfeld: {self.sommerfeld[idx_0]:.3e} ... {self.sommerfeld[idx_1]:.3e}<br>"
            f"Kxx: {self.kxx_interpolated(freq_0):.3e} ... {self.kxx_interpolated(freq_1):.3e} N/m<br>"
            f"Kyy: {self.kyy_interpolated(freq_0):.3e} ... {self.kyy_interpolated(freq_1):.3e} N/m<br>"
            f"Cxx: {self.cxx_interpolated(freq_0):.3e} ... {self.cxx_interpolated(freq_1):.3e} N·s/m<br>"
            f"Cyy: {self.cyy_interpolated(freq_0):.3e} ... {self.cyy_interpolated(freq_1):.3e} N·s/m<br>"
        )

        customdata = [self.n]

        return customdata, hovertemplate


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
    bearing = BearingElement(
        n=0, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2, kzz=1e5, czz=0.5e2, frequency=w
    )
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
