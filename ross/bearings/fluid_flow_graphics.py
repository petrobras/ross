import numpy as np
from plotly import graph_objects as go

from ross.plotly_theme import tableau_colors


def plot_eccentricity(fluid_flow_object, z=0, fig=None, scale_factor=1.0, **kwargs):
    """Plot the rotor eccentricity.

    This function assembles pressure graphic along the z-axis.
    The first few plots are of a different color to indicate where theta begins.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    z: int, optional
        The distance in z where to cut and plot.
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    scale_factor : float, optional
        Scaling factor to plot the rotor shape. Values must range between 0 and 1
        inclusive. 1 is the true scale (1 : 1).
        Default is to 1.0.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> fig = plot_eccentricity(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # fig.show()
    """
    if not 0 <= scale_factor <= 1:
        raise ValueError("scale_factor value must be between 0 and 1.")

    s = 0.5 * scale_factor + 0.5
    angle = fluid_flow_object.attitude_angle
    xre = fluid_flow_object.xre[z]
    xri = fluid_flow_object.xri[z]
    yre = fluid_flow_object.yre[z]
    yri = fluid_flow_object.yri[z]

    val_min = np.min(
        np.sqrt(xre**2 + yre**2) - np.sqrt((xri * s) ** 2 + (yri * s) ** 2)
    )
    val_ref = np.min(np.sqrt(xre**2 + yre**2) - np.sqrt(xri**2 + yri**2))

    if fig is None:
        fig = go.Figure()

    customdata = [
        fluid_flow_object.attitude_angle * 180 / np.pi,
        fluid_flow_object.eccentricity,
        fluid_flow_object.radius_rotor,
        fluid_flow_object.radius_stator,
        fluid_flow_object.preload,
        fluid_flow_object.shape_geometry,
    ]
    hovertemplate_rotor = (
        "Attitude angle: {:.2f}°<br>"
        + "Eccentricity: {:.2e}<br>"
        + "Rotor radius: {:.2e}<br>"
    ).format(customdata[0], customdata[1], customdata[2])
    hovertemplate_stator = (
        "Stator radius: {:.2e}<br>"
        + "Ellipticity ratio: {:.2f}<br>"
        + "Shape Geometry: {}<br>"
    ).format(customdata[3], customdata[4], customdata[5])

    fig.add_trace(
        go.Scatter(
            x=xre,
            y=yre,
            customdata=[customdata] * len(xre),
            mode="markers+lines",
            marker=dict(color=tableau_colors["red"]),
            line=dict(color=tableau_colors["red"]),
            name="Stator",
            legendgroup="Stator",
            hovertemplate=hovertemplate_stator,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xri * s + s * (val_min - val_ref) * np.sin(angle),
            y=yri * s - s * (val_min - val_ref) * np.cos(angle),
            customdata=[customdata] * len(xre),
            mode="markers+lines",
            marker=dict(color=tableau_colors["blue"]),
            line=dict(color=tableau_colors["blue"]),
            name="Rotor",
            legendgroup="Rotor",
            hovertemplate=hovertemplate_rotor,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array([fluid_flow_object.xi]) * s
            + s * (val_min - val_ref) * np.sin(angle),
            y=np.array([fluid_flow_object.yi]) * s
            - s * (val_min - val_ref) * np.cos(angle),
            customdata=[customdata],
            marker=dict(size=8, color=tableau_colors["blue"]),
            name="Rotor",
            legendgroup="Rotor",
            showlegend=False,
            hovertemplate=hovertemplate_rotor,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            customdata=[customdata],
            marker=dict(size=8, color=tableau_colors["red"]),
            name="Stator",
            legendgroup="Stator",
            showlegend=False,
            hovertemplate=hovertemplate_stator,
        )
    )

    fig.update_xaxes(title_text="<b>X axis</b>")
    fig.update_yaxes(title_text="<b>Y axis</b>")
    fig.update_layout(
        title=dict(text="<b>Cut in plane Z={}</b>".format(z)),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        **kwargs,
    )

    return fig


def plot_pressure_z(fluid_flow_object, theta=0, fig=None, **kwargs):
    """Plot the pressure distribution along the z-axis.

    This function assembles pressure graphic along the z-axis for one or both the
    numerically (blue) and analytically (red) calculated pressure matrices, depending
    on if one or both were calculated.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    theta: int, optional
        The theta to be considered.
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_z(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # fig.show()
    """
    if (
        not fluid_flow_object.numerical_pressure_matrix_available
        and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )

    if fig is None:
        fig = go.Figure()
    if fluid_flow_object.numerical_pressure_matrix_available:
        fig.add_trace(
            go.Scatter(
                x=fluid_flow_object.z_list,
                y=fluid_flow_object.p_mat_numerical[:, theta],
                mode="markers+lines",
                line=dict(color=tableau_colors["blue"]),
                showlegend=True,
                name="<b>Numerical pressure</b>",
                hovertemplate=(
                    "<b>Axial Length: %{x:.2f}</b><br>"
                    + "<b>Numerical pressure: %{y:.2f}</b>"
                ),
            )
        )
    if fluid_flow_object.analytical_pressure_matrix_available:
        fig.add_trace(
            go.Scatter(
                x=fluid_flow_object.z_list,
                y=fluid_flow_object.p_mat_analytical[:, theta],
                mode="markers+lines",
                line=dict(color=tableau_colors["red"]),
                showlegend=True,
                name="<b>Analytical pressure</b>",
                hovertemplate=(
                    "<b>Axial Length: %{x:.2f}</b><br>"
                    + "<b>Analytical pressure: %{y:.2f}</b>"
                ),
            )
        )
    fig.update_xaxes(title_text="<b>Axial Length</b>")
    fig.update_yaxes(title_text="<b>Pressure</b>")
    fig.update_layout(
        title=dict(
            text=(
                "<b>Pressure along the flow (axial direction)<b><br>"
                + "<b>Theta={}</b>".format(theta)
            )
        ),
        **kwargs,
    )

    return fig


def plot_shape(fluid_flow_object, theta=0, fig=None, **kwargs):
    """Plot the surface geometry of the rotor.

    This function assembles a graphic representing the geometry of the rotor.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    theta: int, optional
        The theta to be considered.
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> fig = plot_shape(my_fluid_flow, theta=int(my_fluid_flow.ntheta/2))
    >>> # to show the plots you can use:
    >>> # fig.show()
    """
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fluid_flow_object.z_list,
            y=fluid_flow_object.re[:, theta],
            mode="markers+lines",
            line=dict(color=tableau_colors["red"]),
            showlegend=True,
            hoverinfo="none",
            name="<b>Stator</b>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fluid_flow_object.z_list,
            y=fluid_flow_object.ri[:, theta],
            mode="markers+lines",
            line=dict(color=tableau_colors["blue"]),
            showlegend=True,
            hoverinfo="none",
            name="<b>Rotor</b>",
        )
    )
    fig.update_xaxes(title_text="<b>Axial Length</b>")
    fig.update_yaxes(title_text="<b>Radial direction</b>")
    fig.update_layout(
        title=dict(
            text=(
                "<b>Shapes of stator and rotor - Axial direction<b><br>"
                + "<b>Theta={}</b>".format(theta)
            )
        ),
        **kwargs,
    )

    return fig


def plot_pressure_theta(fluid_flow_object, z=0, fig=None, **kwargs):
    """Plot the pressure distribution along theta.

    This function assembles pressure graphic in the theta direction for a given z
    for one or both the numerically (blue) and analytically (red) calculated pressure
    matrices, depending on if one or both were calculated.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    z: int, optional
        The distance along z-axis to be considered.
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # fig.show()
    """
    if (
        not fluid_flow_object.numerical_pressure_matrix_available
        and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if fig is None:
        fig = go.Figure()
    if fluid_flow_object.numerical_pressure_matrix_available:
        fig.add_trace(
            go.Scatter(
                x=fluid_flow_object.gama[z],
                y=fluid_flow_object.p_mat_numerical[z],
                mode="markers+lines",
                line=dict(color=tableau_colors["blue"]),
                showlegend=True,
                name="<b>Numerical pressure</b>",
                hovertemplate=(
                    "<b>Theta: %{x:.2f}</b><br>" + "<b>Numerical pressure: %{y:.2f}</b>"
                ),
            )
        )
    elif fluid_flow_object.analytical_pressure_matrix_available:
        fig.add_trace(
            go.Scatter(
                x=fluid_flow_object.gama[z],
                y=fluid_flow_object.p_mat_analytical[z],
                mode="markers+lines",
                line=dict(color=tableau_colors["red"]),
                showlegend=True,
                name="<b>Analytical pressure</b>",
                hovertemplate=(
                    "<b>Theta: %{x:.2f}</b><br>"
                    + "<b>Analytical pressure: %{y:.2f}</b>"
                ),
            )
        )

    fig.update_xaxes(title_text="<b>Theta value</b>")
    fig.update_yaxes(title_text="<b>Pressure</b>")
    fig.update_layout(
        title=dict(text=("<b>Pressure along Theta | Z={}<b>".format(z))), **kwargs
    )

    return fig


def plot_pressure_theta_cylindrical(
    fluid_flow_object, z=0, from_numerical=True, fig=None, **kwargs
):
    """Plot cylindrical pressure graphic in the theta direction.

    This function assembles cylindrical graphical visualization of the fluid pressure
    in the theta direction for a given axial position (z).

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    z: int, optional
        The distance along z-axis to be considered.
    from_numerical: bool, optional
        If True, takes the numerically calculated pressure matrix as entry.
        If False, takes the analytically calculated one instead.
        If condition cannot be satisfied (matrix not calculated), it will take the one
        that is available and raise a warning.
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_theta_cylindrical(my_fluid_flow, z=int(my_fluid_flow.nz/2))
    >>> # to show the plots you can use:
    >>> # fig.show()
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

    r = np.linspace(fluid_flow_object.radius_rotor, fluid_flow_object.radius_stator)
    theta = np.linspace(
        0.0, 2.0 * np.pi + fluid_flow_object.dtheta / 2, fluid_flow_object.ntheta
    )
    theta *= 180 / np.pi

    pressure_along_theta = p_mat[z, :]
    min_pressure = np.amin(pressure_along_theta)

    r_matrix, theta_matrix = np.meshgrid(r, theta)
    z_matrix = np.zeros((theta.size, r.size))

    for i in range(0, theta.size):
        inner_radius = np.sqrt(
            fluid_flow_object.xri[z][i] * fluid_flow_object.xri[z][i]
            + fluid_flow_object.yri[z][i] * fluid_flow_object.yri[z][i]
        )

        for j in range(r.size):
            if r_matrix[i][j] < inner_radius:
                continue
            z_matrix[i][j] = pressure_along_theta[i] - min_pressure + 0.01

    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=r_matrix.ravel(),
            theta=theta_matrix.ravel(),
            customdata=z_matrix.ravel(),
            marker=dict(
                color=z_matrix.ravel(),
                colorscale="Viridis",
                cmin=np.amin(z_matrix),
                cmax=np.amax(z_matrix),
                colorbar=dict(title=dict(text="<b>Pressure</b>", side="top")),
            ),
            thetaunit="degrees",
            name="Pressure",
            showlegend=False,
            hovertemplate=(
                "<b>Raddi: %{r:.4e}</b><br>"
                + "<b>θ: %{theta:.2f}</b><br>"
                + "<b>Pressure: %{customdata:.4e}</b>"
            ),
        )
    )
    fig.update_layout(
        polar=dict(
            hole=0.5,
            bargap=0.0,
            angularaxis=dict(
                rotation=-90 - fluid_flow_object.attitude_angle * 180 / np.pi
            ),
        ),
        **kwargs,
    )
    return fig


def plot_pressure_surface(fluid_flow_object, fig=None, **kwargs):
    """Assembles pressure surface graphic in the bearing, using Plotly.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    kwargs : optional
        Additional key word arguments can be passed to change the plot layout only
        (e.g. width=1000, height=800, ...).
        *See Plotly Python Figure Reference for more information.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.

    Examples
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> fig = plot_pressure_surface(my_fluid_flow)
    >>> # to show the plots you can use:
    >>> # fig.show()
    """
    if (
        not fluid_flow_object.numerical_pressure_matrix_available
        and not fluid_flow_object.analytical_pressure_matrix_available
    ):
        raise ValueError(
            "Must calculate the pressure matrix. "
            "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
        )
    if fig is None:
        fig = go.Figure()
    if fluid_flow_object.numerical_pressure_matrix_available:
        z, theta = np.meshgrid(fluid_flow_object.z_list, fluid_flow_object.gama[0])
        fig.add_trace(
            go.Surface(
                x=z,
                y=theta,
                z=fluid_flow_object.p_mat_numerical.T,
                colorscale="Viridis",
                cmin=np.amin(fluid_flow_object.p_mat_numerical.T),
                cmax=np.amax(fluid_flow_object.p_mat_numerical.T),
                colorbar=dict(title=dict(text="<b>Pressure</b>", side="top")),
                name="Pressure",
                showlegend=False,
                hovertemplate=(
                    "<b>Length: %{x:.2e}</b><br>"
                    + "<b>Angular Position: %{y:.2f}</b><br>"
                    + "<b>Pressure: %{z:.2f}</b>"
                ),
            )
        )

    fig.update_layout(
        scene=dict(
            bgcolor="white",
            xaxis=dict(title=dict(text="<b>Rotor Length</b>")),
            yaxis=dict(title=dict(text="<b>Angular Position</b>")),
            zaxis=dict(title=dict(text="<b>Pressure</b>")),
        ),
        title=dict(text="<b>Bearing Pressure Field</b>"),
        **kwargs,
    )

    return fig
