
import ross as rs
from ross.rotor_assembly import Rotor
from ross.gear_mesh_TVMS import GearElementTVMS, Mesh
from ross.multi_rotor_TVMS import MultiRotorTVMS
import time
from ross.units import Q_
import plotly.io as pio
import numpy as np
import os

def two_shaft_rotor_example(run_type: str):
    """Create a multi-rotor as example.

    This function returns an instance of two-shaft rotor system from Rao et al.
    This typical example is a turbo-alternator rotor system, which consists of
    a generator rotor, a turbine rotor and a spur gear pair connecting two rotors.
    Each rotor is supported by a pair of bearing two shaft elements, one disk and
    two simple bearings.

    The purpose of this is to make available a simple model so that doctest can
    be written using this.

    Returns
    -------
    An instance of a rotor object.

    References
    ----------
    Rao, J. S., Shiau, T. N., chang, J. R. (1998). Theoretical analysis of lateral
    response due to torsional excitation of geared rotors. Mechanism and Machine Theory,
    33 (6), 761-783. doi: 10.1016/S0094-114X(97)00056-6

    Examples
    --------
    >>> multi_rotor = two_shaft_rotor_example()
    >>> modal = multi_rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([ 74.,  77., 112., 113.])
    """
    # A spur geared two-shaft rotor system.
    material = rs.Material(name="mat_steel", rho=7800, E=207e9, G_s=79.5e9)

    # Rotor 1
    L1 = [0.1, 4.24, 1.16, 0.3]
    d1 = [0.3, 0.3, 0.22, 0.22]
    shaft1 = [
        rs.ShaftElement(
            L=L1[i],
            idl=0.0,
            odl=d1[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L1))
    ]

    generator = rs.DiskElement(
        n=1,
        m=525.7,
        Id=16.1,
        Ip=32.2,
    )
    disk = rs.DiskElement(
        n=2,
        m=116.04,
        Id=3.115,
        Ip=6.23,
    )
    
    gear1 = GearElementTVMS(n=4, m=5, module=Q_(2,'mm'), width=Q_(2, 'cm'), n_tooth=40, hub_bore_radius=Q_(4, 'cm'))

    bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)

    rotor1 = rs.Rotor(
        shaft1,
        [generator, disk, gear1],
        [bearing1, bearing2],
    )

    # Rotor 2
    L2 = [0.3, 5, 0.1]
    d2 = [0.15, 0.15, 0.15]
    shaft2 = [
        rs.ShaftElement(
            L=L2[i],
            idl=0.0,
            odl=d2[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L2))
    ]
    
    gear2 = GearElementTVMS(n=0, m=Q_(6, 'kg'), module=Q_(2, 'mm'), width=Q_(2,'cm'), n_tooth=75, hub_bore_radius=Q_(7.5,'cm'))

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )

    if run_type == 'tvms':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            tvms=True,
        )

    if run_type == 'max_stiffness':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            only_max_stiffness=True,
        )

    if run_type == 'user_defined':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            user_defined_stiffness=4e8
        )

def multirotor_run(t=10, speed=50, run_type = 'tvms', dt = 1e-5, unb_mag = [35e-4, 40e-4]) -> None:
    """
    Run a time response simulation of a two-shaft geared multi-rotor system.

    This function performs the time-domain simulation of the lateral response of
    a multi-rotor system composed of a generator rotor and a turbine rotor
    connected via a spur gear. The system is excited by unbalance forces applied
    to specified nodes of the rotor structure, and the response is computed using
    a time integration method (Newmark).

    The multi-rotor model is generated using the `two_shaft_rotor_example` function,
    and the type of time-varying mesh stiffness (TVMS) can be configured by the
    `run_type` parameter. The simulation tracks the lateral vibration at a specific
    node of interest (hardcoded as node 3), storing its time-history in the
    horizontal and vertical directions.

    Parameters
    ----------
    id : str
        Identifier for the simulation run, used for logging purposes.
    t : float, optional
        Final simulation time in seconds. Default is 10.
    speed : float, optional
        Rotational speed of the generator rotor in Hz. Default is 50.
    run_type : str, optional
        Type of gear coupling modeling to use. Options include:
            - 'tvms' : smoothly interpolated TVMS.
            - 'max_stiffness' : maximum stiffness used as constant.
            - 'TVMS'          : true time-varying mesh stiffness function.
            - 'user_defined'  : user-defined constant mesh stiffness.
        Default is 'tvms'.
    dt : float, optional
        Time step for numerical integration. Default is 1e-5 seconds.
    unb_mag : list of float, optional
        Magnitudes of unbalance forces (in m·kg) applied to the selected nodes.
        Should be a list of two elements corresponding to each unbalanced node.
        Default is [35e-4, 40e-4].

    Returns
    -------
    None
        This function runs the simulation and prints progress and information
        to the console. The displacement response at a specified node is stored
        internally and can be plotted or saved by modifying the function.

    Examples
    --------
    >>> multirotor_run(id="test01", t=5, speed=60, run_type='TVMS', dt=1e-4, unb_mag=[20e-4, 25e-4])
    [PID 12345] Iniciando Simulação: test01 (Speed: 60 Hz, run_type: TVMS, dt: 0.0001, t_final: 5), unb: [0.002, 0.0025]

    Notes
    -----
    The unbalance forces are modeled as rotating vectors whose phase evolves
    with time according to the local speed of each rotor node. This simulation
    assumes linear bearing behavior and does not include nonlinearities or thermal effects.

    References
    ----------
    Rao, J. S., Shiau, T. N., Chang, J. R. (1998). Theoretical analysis of lateral
    response due to torsional excitation of geared rotors. Mechanism and Machine Theory,
    33(6), 761-783. https://doi.org/10.1016/S0094-114X(97)00056-6
    """

    print(f"Iniciando Simulação: (Speed: {speed} Hz, run_type: {run_type}, dt: {dt}, t_final: {t}), unb: {unb_mag}")
    
    run_type = run_type
    rotor = two_shaft_rotor_example(run_type=run_type)

    nodes = [2, 7]
    unb_mag = unb_mag
    unb_phase = [0, 0]

    dt = dt
    t = np.arange(0, t, dt)
    speed1 = speed*2*np.pi  # Generator rotor speed

    num_dof = rotor.number_dof

    F = np.zeros((len(t), rotor.ndof))

    for i, node in enumerate(nodes):
        speed = rotor.check_speed(node, speed1)
        phi = speed * t + unb_phase[i]

        dofx = num_dof * node + 0
        dofy = num_dof * node + 1
        F[:, dofx] += unb_mag[i] * (speed**2) * np.cos(phi)
        F[:, dofy] += unb_mag[i] * (speed**2) * np.sin(phi)

    tr = rotor.run_time_response(speed1, F, t, method='newmark', progress_interval=0.1)

    node = 3
    dof_node_x = node * rotor.number_dof + 5
    dof_node_y = dof_node_x + 1
    
    x = tr.yout[:,dof_node_x]
    y = tr.yout[:,dof_node_y]
    
multirotor_run(run_type='user_defined')