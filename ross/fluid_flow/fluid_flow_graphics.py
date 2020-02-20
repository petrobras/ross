import matplotlib.pyplot as plt
from bokeh.plotting import figure
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def plot_eccentricity(fluid_flow_object, z=0):
    """This function assembles pressure graphic along the z-axis.
    The first few plots are of a different color to indicate where theta begins.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    z: int, optional
        The distance in z where to cut and plot.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> fig = plot_eccentricity(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    p = figure(
        title="Cut in plane Z=" + str(z),
        x_axis_label="X axis",
        y_axis_label="Y axis",
    )
    for j in range(0, fluid_flow_object.ntheta):
        p.circle(fluid_flow_object.xre[z][j], fluid_flow_object.yre[z][j], color="red")
        p.circle(fluid_flow_object.xri[z][j], fluid_flow_object.yri[z][j], color="blue")
        p.circle(0, 0, color="blue")
        p.circle(fluid_flow_object.xi, fluid_flow_object.yi, color="red")
    p.circle(0, 0, color="black")
    return p


def plot_pressure_z(fluid_flow_object, theta=0):
    """This function assembles pressure graphic along the z-axis for one or both the
    numerically (blue) and analytically (red) calculated pressure matrices, depending on if
    one or both were calculated.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    theta: int, optional
        The theta to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_z(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    y_n = []
    y_a = []
    for i in range(0, fluid_flow_object.nz):
        y_n.append(fluid_flow_object.p_mat_numerical[i][theta])
        y_a.append(fluid_flow_object.p_mat_analytical[i][theta])
    p = figure(
        title="Pressure along the Z direction (direction of flow); Theta="
              + str(theta),
        x_axis_label="Points along Z",
    )
    if fluid_flow_object.numerical_pressure_matrix_available:
        p.line(fluid_flow_object.z_list, y_n, legend="Numerical pressure", color="blue", line_width=2)
    if fluid_flow_object.analytical_pressure_matrix_available:
        p.line(fluid_flow_object.z_list, y_a, legend="Analytical pressure", color="red", line_width=2)
    return p


def plot_shape(fluid_flow_object, theta=0):
    """This function assembles a graphic representing the geometry of the rotor.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    theta: int, optional
        The theta to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> fig = plot_shape(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    y_re = np.zeros(fluid_flow_object.nz)
    y_ri = np.zeros(fluid_flow_object.nz)
    for i in range(0, fluid_flow_object.nz):
        y_re[i] = fluid_flow_object.re[i][theta]
        y_ri[i] = fluid_flow_object.ri[i][theta]
    p = figure(
        title="Shapes of stator and rotor along Z; Theta=" + str(theta),
        x_axis_label="Points along Z",
        y_axis_label="Radial direction",
    )
    p.line(fluid_flow_object.z_list, y_re, line_width=2, color="red")
    p.line(fluid_flow_object.z_list, y_ri, line_width=2, color="blue")
    return p


def plot_pressure_theta(fluid_flow_object, z=0):
    """This function assembles pressure graphic in the theta direction for a given z
    for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
    depending on if one or both were calculated.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    z: int, optional
        The distance along z-axis to be considered.
    Returns
    -------
    Figure
        An object containing the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # show(fig)
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
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
    if fluid_flow_object.numerical_pressure_matrix_available:
        p.line(
            fluid_flow_object.gama[z],
            fluid_flow_object.p_mat_numerical[z],
            legend="Numerical pressure",
            line_width=2,
            color="blue",
        )
    elif fluid_flow_object.analytical_pressure_matrix_available:
        p.line(
            fluid_flow_object.gama[z],
            fluid_flow_object.p_mat_analytical[z],
            legend="Analytical pressure",
            line_width=2,
            color="red",
        )
    return p


def matplot_eccentricity(fluid_flow_object, z=0, ax=None):
    """This function assembles pressure graphic along the z-axis using matplotlib.
    The first few plots are of a different color to indicate where theta begins.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> fig, ax = plt.subplots()
    >>> ax = matplot_eccentricity(my_fluid_flow, z=int(my_fluid_flow.nz/2), ax=ax)
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if ax is None:
        ax = plt.gca()
    x_r = []
    x_b = []
    y_r = []
    y_b = []
    for j in range(0, fluid_flow_object.ntheta):
        x_r.append(fluid_flow_object.xre[z][j])
        y_r.append(fluid_flow_object.yre[z][j])
        x_b.append(fluid_flow_object.xri[z][j])
        y_b.append(fluid_flow_object.yri[z][j])
    ax.plot(x_r, y_r, "r.")
    ax.plot(x_b, y_b, "b.")
    ax.plot(0, 0, "r*")
    ax.plot(fluid_flow_object.xi, fluid_flow_object.yi, "b*")
    ax.set_title("Cut in plane Z=" + str(z))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.axis("equal")
    return ax


def matplot_pressure_z(fluid_flow_object, theta=0, ax=None):
    """This function assembles pressure graphic along the z-axis using matplotlib
    for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
    depending on if one or both were calculated.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_z(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if ax is None:
        ax = plt.gca()
    y_n = np.zeros(fluid_flow_object.nz)
    y_a = np.zeros(fluid_flow_object.nz)
    for i in range(0, fluid_flow_object.nz):
        y_n[i] = fluid_flow_object.p_mat_numerical[i][theta]
        y_a[i] = fluid_flow_object.p_mat_analytical[i][theta]
    if fluid_flow_object.numerical_pressure_matrix_available:
        ax.plot(fluid_flow_object.z_list, y_n, "b", label="Numerical pressure")
    if fluid_flow_object.analytical_pressure_matrix_available:
        ax.plot(fluid_flow_object.z_list, y_a, "r", label="Analytical pressure")
    ax.set_title(
        "Pressure along the Z direction (direction of flow); Theta=" + str(theta)
    )
    ax.set_xlabel("Points along Z")
    ax.set_ylabel("Pressure")
    return ax


def matplot_shape(fluid_flow_object, theta=0, ax=None):
    """This function assembles a graphic representing the geometry of the rotor using matplotlib.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> ax = matplot_shape(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if ax is None:
        ax = plt.gca()
    y_ext = np.zeros(fluid_flow_object.nz)
    y_int = np.zeros(fluid_flow_object.nz)
    for i in range(0, fluid_flow_object.nz):
        y_ext[i] = fluid_flow_object.re[i][theta]
        y_int[i] = fluid_flow_object.ri[i][theta]
    ax.plot(fluid_flow_object.z_list, y_ext, "r")
    ax.plot(fluid_flow_object.z_list, y_int, "b")
    ax.set_title("Shapes of stator and rotor along Z; Theta=" + str(theta))
    ax.set_xlabel("Points along Z")
    ax.set_ylabel("Radial direction")
    return ax


def matplot_pressure_theta_cylindrical(fluid_flow_object, z=0, from_numerical=True, ax=None):
    """This function assembles cylindrical pressure graphic in the theta direction for a given z,
    using matplotlib.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_theta_cylindrical(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if from_numerical:
        if fluid_flow_object.numerical_pressure_matrix_available:
            p_mat = fluid_flow_object.p_mat_numerical
        else:
            p_mat = fluid_flow_object.p_mat_analytical
    else:
        if fluid_flow_object.analytical_pressure_matrix_available:
            p_mat = fluid_flow_object.p_mat_analytical
        else:
            p_mat = fluid_flow_object.p_mat_numerical
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    r = np.arange(
        0,
        fluid_flow_object.radius_stator + 0.0001,
        (fluid_flow_object.radius_stator - fluid_flow_object.radius_rotor) / fluid_flow_object.nradius,
    )
    theta = np.arange(-np.pi * 0.25, 1.75 * np.pi + fluid_flow_object.dtheta / 2, fluid_flow_object.dtheta)

    pressure_along_theta = np.zeros(fluid_flow_object.ntheta)
    for i in range(0, fluid_flow_object.ntheta):
        pressure_along_theta[i] = p_mat[0][i]

    min_pressure = np.amin(pressure_along_theta)

    r_matrix, theta_matrix = np.meshgrid(r, theta)
    z_matrix = np.zeros((theta.size, r.size))
    inner_radius_list = np.zeros(fluid_flow_object.ntheta)
    pressure_list = np.zeros((theta.size, r.size))
    for i in range(0, theta.size):
        inner_radius = np.sqrt(
            fluid_flow_object.xri[z][i] * fluid_flow_object.xri[z][i] + fluid_flow_object.yri[z][i] * fluid_flow_object.yri[z][i]
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


def matplot_pressure_theta(fluid_flow_object, z=0, ax=None):
    """This function assembles pressure graphic in the theta direction for a given z,
    using matplotlib.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if ax is None:
        ax = plt.gca()
    if fluid_flow_object.numerical_pressure_matrix_available:
        ax.plot(
            fluid_flow_object.gama[z],
            fluid_flow_object.p_mat_numerical[z],
            "b",
            label="Numerical pressure"
        )
    if fluid_flow_object.analytical_pressure_matrix_available:
        ax.plot(
            fluid_flow_object.gama[z],
            fluid_flow_object.p_mat_analytical[z],
            "r",
            label="Analytical pressure",
        )
    ax.set_title("Pressure along Theta; Z=" + str(z))
    ax.set_xlabel("Points along Theta")
    ax.set_ylabel("Pressure")
    return ax


def matplot_pressure_surface(fluid_flow_object, ax=None):
    """This function assembles pressure surface graphic in the bearing, using matplotlib.
    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.
    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> ax = matplot_pressure_surface(my_fluid_flow)
    >>> # to show the plots you can use:
    >>> # plt.show()
    """
    if (
            not fluid_flow_object.numerical_pressure_matrix_available
            and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    if fluid_flow_object.numerical_pressure_matrix_available:
        z, theta = np.meshgrid(fluid_flow_object.z_list, fluid_flow_object.gama[0])
        ax.plot_surface(
            z, theta, fluid_flow_object.p_mat_numerical.T, cmap=cm.coolwarm, linewidth=0,
            label="Numerical pressure"
        )
    if fluid_flow_object.analytical_pressure_matrix_available:
        z, theta = np.meshgrid(fluid_flow_object.z_list, fluid_flow_object.gama[0])
        ax.plot_surface(
            z, theta, fluid_flow_object.p_mat_analytical.T, cmap=cm.coolwarm, linewidth=0,
            label="Analytical pressure"
        )
    ax.set_title('Bearing Pressure Field', fontsize=18)
    ax.set_xlabel('Bearing Length', fontsize=14, linespacing=50)
    ax.set_ylabel('Angular position', fontsize=14, linespacing=50)
    ax.set_zlabel('Pressure', fontsize=14, linespacing=50)
    ax.dist = 10
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    return ax

