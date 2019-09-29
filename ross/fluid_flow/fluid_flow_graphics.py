import matplotlib.pyplot as plt
from bokeh.plotting import figure
import numpy as np


def plot_eccentricity(pressure_matrix_object, z=0):
    """This function assembles pressure graphic along the z-axis.
    The first few plots are of a different color to indicate where theta begins.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    z: int, optional
        The distance in z where to cut and plot.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> fig = plot_eccentricity(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    p = figure(
        title="Cut in plane Z=" + str(z),
        x_axis_label="X axis",
        y_axis_label="Y axis",
    )
    for j in range(0, pressure_matrix_object.ntheta):
        p.circle(pressure_matrix_object.xre[z][j], pressure_matrix_object.yre[z][j], color="red")
        p.circle(pressure_matrix_object.xri[z][j], pressure_matrix_object.yri[z][j], color="blue")
        p.circle(0, 0, color="blue")
        p.circle(pressure_matrix_object.xi, pressure_matrix_object.yi, color="red")
    p.circle(0, 0, color="black")
    return p


def plot_pressure_z(pressure_matrix_object, theta=0):
    """This function assembles pressure graphic along the z-axis for one or both the
    numerically (blue) and analytically (red) calculated pressure matrices, depending on if
    one or both were calculated.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    theta: int, optional
        The theta to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_z(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    if (
            not pressure_matrix_object.numerical_pressure_matrix_available
            and not pressure_matrix_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    y_n = []
    y_a = []
    for i in range(0, pressure_matrix_object.nz):
        y_n.append(pressure_matrix_object.p_mat_numerical[i][theta])
        y_a.append(pressure_matrix_object.p_mat_analytical[i][theta])
    p = figure(
        title="Pressure along the Z direction (direction of flow); Theta="
              + str(theta),
        x_axis_label="Points along Z",
    )
    if pressure_matrix_object.numerical_pressure_matrix_available:
        p.line(pressure_matrix_object.z_list, y_n, legend="Numerical pressure", color="blue", line_width=2)
    if pressure_matrix_object.analytical_pressure_matrix_available:
        p.line(pressure_matrix_object.z_list, y_a, legend="Analytical pressure", color="red", line_width=2)
    return p


def plot_shape(pressure_matrix_object, theta=0):
    """This function assembles a graphic representing the geometry of the rotor.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    theta: int, optional
        The theta to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> fig = plot_shape(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    y_re = np.zeros(pressure_matrix_object.nz)
    y_ri = np.zeros(pressure_matrix_object.nz)
    for i in range(0, pressure_matrix_object.nz):
        y_re[i] = pressure_matrix_object.re[i][theta]
        y_ri[i] = pressure_matrix_object.ri[i][theta]
    p = figure(
        title="Shapes of stator and rotor along Z; Theta=" + str(theta),
        x_axis_label="Points along Z",
        y_axis_label="Radial direction",
    )
    p.line(pressure_matrix_object.z_list, y_re, line_width=2, color="red")
    p.line(pressure_matrix_object.z_list, y_ri, line_width=2, color="blue")
    return p


def plot_pressure_theta(pressure_matrix_object, z=0):
    """This function assembles pressure graphic in the theta direction for a given z
    for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
    depending on if one or both were calculated.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    z: int, optional
        The distance along z-axis to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    if (
            not pressure_matrix_object.numerical_pressure_matrix_available
            and not pressure_matrix_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    p = figure(
        title="Pressure along Theta; Z=" + str(z),
        x_axis_label="Points along Theta",
        y_axis_label="Pressure",
    )
    if pressure_matrix_object.numerical_pressure_matrix_available:
        p.line(
            pressure_matrix_object.gama[z],
            pressure_matrix_object.p_mat_numerical[z],
            legend="Numerical pressure",
            line_width=2,
            color="blue",
        )
    elif pressure_matrix_object.analytical_pressure_matrix_available:
        p.line(
            pressure_matrix_object.gama[z],
            pressure_matrix_object.p_mat_analytical[z],
            legend="Analytical pressure",
            line_width=2,
            color="red",
        )
    return p


def matplot_eccentricity(pressure_matrix_object, z=0, ax=None):
    """This function assembles pressure graphic along the z-axis using matplotlib.
    The first few plots are of a different color to indicate where theta begins.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    z: int, optional
        The distance in z where to cut and plot.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> ax = matplot_eccentricity(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if ax is None:
        ax = plt.gca()
    for j in range(0, pressure_matrix_object.ntheta):
        ax.plot(pressure_matrix_object.xre[z][j], pressure_matrix_object.yre[z][j], "r.")
        ax.plot(pressure_matrix_object.xri[z][j], pressure_matrix_object.yri[z][j], "b.")
        ax.plot(0, 0, "r*")
        ax.plot(pressure_matrix_object.xi, pressure_matrix_object.yi, "b*")
    ax.set_title("Cut in plane Z=" + str(z))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.axis("equal")
    return ax


def matplot_pressure_z(pressure_matrix_object, theta=0, ax=None):
    """This function assembles pressure graphic along the z-axis using matplotlib
    for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
    depending on if one or both were calculated.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    theta: int, optional
        The distance in theta where to cut and plot.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_z(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not pressure_matrix_object.numerical_pressure_matrix_available
            and not pressure_matrix_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if ax is None:
        ax = plt.gca()
    y_n = np.zeros(pressure_matrix_object.nz)
    y_a = np.zeros(pressure_matrix_object.nz)
    for i in range(0, pressure_matrix_object.nz):
        y_n[i] = pressure_matrix_object.p_mat_numerical[i][theta]
        y_a[i] = pressure_matrix_object.p_mat_analytical[i][theta]
    if pressure_matrix_object.numerical_pressure_matrix_available:
        ax.plot(pressure_matrix_object.z_list, y_n, "b", label="Numerical pressure")
    if pressure_matrix_object.analytical_pressure_matrix_available:
        ax.plot(pressure_matrix_object.z_list, y_a, "r", label="Analytical pressure")
    ax.set_title(
        "Pressure along the Z direction (direction of flow); Theta=" + str(theta)
    )
    ax.set_xlabel("Points along Z")
    ax.set_ylabel("Pressure")
    return ax


def matplot_shape(pressure_matrix_object, theta=0, ax=None):
    """This function assembles a graphic representing the geometry of the rotor using matplotlib.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    theta: int, optional
        The theta to be considered.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> ax = matplot_shape(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if ax is None:
        ax = plt.gca()
    y_ext = np.zeros(pressure_matrix_object.nz)
    y_int = np.zeros(pressure_matrix_object.nz)
    for i in range(0, pressure_matrix_object.nz):
        y_ext[i] = pressure_matrix_object.re[i][theta]
        y_int[i] = pressure_matrix_object.ri[i][theta]
    ax.plot(pressure_matrix_object.z_list, y_ext, "r")
    ax.plot(pressure_matrix_object.z_list, y_int, "b")
    ax.set_title("Shapes of stator and rotor along Z; Theta=" + str(theta))
    ax.set_xlabel("Points along Z")
    ax.set_ylabel("Radial direction")
    return ax


def matplot_pressure_theta_cylindrical(pressure_matrix_object, z=0, from_numerical=True, ax=None):
    """This function assembles cylindrical pressure graphic in the theta direction for a given z,
    using matplotlib.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    z: int, optional
        The distance along z-axis to be considered.
    from_numerical: bool, optional
        If True, takes the numerically calculated pressure matrix as entry.
        If False, takes the analytically calculated one instead.
        If condition cannot be satisfied (matrix not calculated), it will take the one that is available
        and raise a warning.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_theta_cylindrical(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not pressure_matrix_object.numerical_pressure_matrix_available
            and not pressure_matrix_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if from_numerical:
        if pressure_matrix_object.numerical_pressure_matrix_available:
            p_mat = pressure_matrix_object.p_mat_numerical
        else:
            p_mat = pressure_matrix_object.p_mat_analytical
    else:
        if pressure_matrix_object.analytical_pressure_matrix_available:
            p_mat = pressure_matrix_object.p_mat_analytical
        else:
            p_mat = pressure_matrix_object.p_mat_numerical
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    r = np.arange(
        0,
        pressure_matrix_object.radius_stator + 0.0001,
        (pressure_matrix_object.radius_stator - pressure_matrix_object.radius_rotor) / pressure_matrix_object.nradius,
    )
    theta = np.arange(-np.pi * 0.25, 1.75 * np.pi + pressure_matrix_object.dtheta / 2, pressure_matrix_object.dtheta)

    pressure_along_theta = np.zeros(pressure_matrix_object.ntheta)
    for i in range(0, pressure_matrix_object.ntheta):
        pressure_along_theta[i] = p_mat[0][i]

    min_pressure = np.amin(pressure_along_theta)

    r_matrix, theta_matrix = np.meshgrid(r, theta)
    z_matrix = np.zeros((theta.size, r.size))
    inner_radius_list = np.zeros(pressure_matrix_object.ntheta)
    pressure_list = np.zeros((theta.size, r.size))
    for i in range(0, theta.size):
        inner_radius = np.sqrt(
            pressure_matrix_object.xri[z][i] * pressure_matrix_object.xri[z][i] + pressure_matrix_object.yri[z][i] * pressure_matrix_object.yri[z][i]
        )
        inner_radius_list[i] = inner_radius
        for j in range(0, r.size):
            if r_matrix[i][j] < inner_radius:
                continue
            pressure_list[i][j] = pressure_along_theta[i]
            z_matrix[i][j] = pressure_along_theta[i] - min_pressure + 0.01
    ax.contourf(theta_matrix, r_matrix, z_matrix, cmap="coolwarm")
    ax.set_title("Pressure along Theta; Z=" + str(z))
    return ax


def matplot_pressure_theta(pressure_matrix_object, z=0, ax=None):
    """This function assembles pressure graphic in the theta direction for a given z,
    using matplotlib.
    Parameters
    ----------
    pressure_matrix_object: a PressureMatrix object
    z: int, optional
        The distance along z-axis to be considered.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not pressure_matrix_object.numerical_pressure_matrix_available
            and not pressure_matrix_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if ax is None:
        ax = plt.gca()
    if pressure_matrix_object.numerical_pressure_matrix_available:
        ax.plot(
            pressure_matrix_object.gama[z],
            pressure_matrix_object.p_mat_numerical[z],
            "b",
            label="Numerical pressure"
        )
    if pressure_matrix_object.analytical_pressure_matrix_available:
        ax.plot(
            pressure_matrix_object.gama[z],
            pressure_matrix_object.p_mat_analytical[z],
            "r",
            label="Analytical pressure",
        )
    ax.set_title("Pressure along Theta; Z=" + str(z))
    ax.set_xlabel("Points along Theta")
    ax.set_ylabel("Pressure")
    return ax


