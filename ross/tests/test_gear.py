import numpy as np
from plotly import graph_objects as go
from ross.units import Q_
from ross.materials import Material
import pandas as pd
import scipy as sp
from ross.gear_mesh_TVMS import GearElementTVMS, Mesh
from ross.disk_element import DiskElement

def gear_mesh_compare() -> None:
    """
    Evaluate and plot the equivalent mesh stiffness of a gear pair consisting of two 62-tooth gears.

    This function creates two identical gears with specified material and geometric properties,
    simulates their time-varying mesh stiffness over two full mesh cycles, and computes the equivalent
    stiffness along discrete time steps.

    The resulting equivalent mesh stiffness is plotted against the mesh angle in degrees,
    illustrating the periodic variation caused by the changing contact conditions between the gear teeth.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function displays an interactive plot of gear mesh stiffness versus mesh angle.

    Notes
    -----
    The mesh stiffness calculation assumes rigid body speed and gear geometry remain constant during the simulation.
    The mesh cycles are calculated based on the angular velocity and number of teeth.
    """
    gear_material_ma = Material("ma_steel", rho=Q_(7.81, 'g/cm**3'), E=Q_(206, 'GPa'), Poisson=0.3)
    
    gear1 = GearElementTVMS(n=21, material=gear_material_ma, m=Q_(12*2.204,'lbs'), module=Q_(2, 'mm'), width=Q_(20,'mm'), n_tooth=62, hub_bore_radius=Q_(17.5,'mm'), pr_angle=Q_(20,'deg'))
    gear2 = GearElementTVMS(n=21, material=gear_material_ma, m=Q_(12*2.204,'lbs'), module=Q_(2,'mm'), width=Q_(20,'mm'), n_tooth=62, hub_bore_radius=Q_(17.5, 'mm'), pr_angle=Q_(20,'deg'))

    n_tm = 2

    gear1Speed = 10 * 2 * np.pi

    meshing = Mesh(gear1, gear2, interpolation=False)   
    
    dt      = 2 * np.pi / (100 * gear1Speed * gear1.n_tooth)

    time_range  = np.arange(0, n_tm * 2 * np.pi / (gear1Speed * gear1.n_tooth), dt)

    speed_range = gear1Speed * np.ones(np.shape(time_range))

    stiffness = np.zeros(np.shape(time_range))
    k0_stiffness = np.zeros(np.shape(time_range))
    k1_stiffness = np.zeros(np.shape(time_range))

    for i, time in enumerate(time_range):
        time = float(time)
        speed = float(speed_range[i])
        stiffness[i], k0_stiffness[i], k1_stiffness[i] = meshing.mesh(speed, time)


    standard_font = dict(size=15, color="black", weight="bold")
    axis_font = dict(size=25, color="black", weight="bold")

    # Calculate limits and yticks
    x_lim = time_range[-1]
    # yticks = np.arange(3.8e8, int(4.4e8), int(0.1e8))

    # Create figure
    fig = go.Figure()

    # Add the main plot lines
    fig.add_trace(go.Scatter(
        x=time_range*gear1Speed*360/2/np.pi,
        y=stiffness,
        mode='lines',
        line=dict(color='red', width=3),
        name='Stiffness'
    ))

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title='Angle [degree]',

            range=[0, x_lim*gear1Speed*360/2/np.pi],
        ),
        yaxis=dict(
            title='Stiffness [N/m]',
            range=[2e8, 5e8],
            tickformat=".1e",  # Use scientific notation for y-axis labels
        ),
        font=axis_font,  # Customize font size and color
        showlegend=True
    )

    fig.show()


def gear_stiffness_example():
    """
    Compare and plot the contributions of different stiffness components to the equivalent gear mesh stiffness.

    This function computes stiffness components (ka, kb, kf, ks) as functions of the mesh angle for a single gear,
    representing different physical contributions such as bending, base tooth stiffness, axial and shear stresses.

    The stiffness components are plotted over the mesh angle range, providing insight into their relative
    magnitudes and how each contributes to the overall gear mesh stiffness variation.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Displays a plot comparing stiffness components against mesh angle.

    Notes
    -----
    The angular range is taken from the gear's critical contact angle to the addendum angle,
    with stiffness calculated through the gear's internal method vectorized over this range.
    """
    
    gear1               = GearElementTVMS(n=21, m=Q_(12*2.204,'lbs'), module=Q_(2, 'mm'), width=Q_(2, 'cm'), n_tooth=55, hub_bore_radius=Q_(17.5, 'mm'))
    computeStiffness    = np.vectorize(gear1._compute_stiffness)

    angle_range = np.linspace(gear1.geometry_dict['alpha_c'], gear1.geometry_dict['alpha_a'], 200)
    ka, kb, kf, ks = computeStiffness(angle_range)
    
    # Create the figure
    fig = go.Figure()

    stiffness_dict = {"ka": ka, "kb": kb, "kf": kf, "ks": ks}

    # Add traces for each stiffness component
    for name, values in stiffness_dict.items():
        fig.add_trace(go.Scatter(x=angle_range*180/np.pi, y=values, mode='lines', name=name))

    # Customize layout
    fig.update_layout(
        title="Stiffness Variation vs. Angle",
        xaxis_title="Angle (radians or degrees)",  # Adjust accordingly
        yaxis_title="Stiffness",
        template="plotly_dark",  # Optional: Choose from 'plotly', 'plotly_dark', etc.
        legend_title="Stiffness Components",
        yaxis_tickformat='.2e',
    )

    # Show plot
    fig.show()

gear_mesh_compare()