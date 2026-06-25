from ross import (
    MagneticBearingElement,
    Material,
    ShaftElement,
    DiskElement,
    Rotor,
)

import numpy as np
import control as ct

from ross.bearings.magnetic.amb_controllers import s


def rotor_example_amb_complex_controllers(ambs=True):
    """
    Create a rotor example with complex magnetic bearing controllers.

    Parameters
    ----------
    ambs : bool, optional
        Whether to include magnetic bearing elements in the rotor model.
        Defaults to True.

    Returns
    -------
    rotor : ross.Rotor
        The constructed rotor model.

    Examples
    --------
    >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
    >>> rotor_amb = rotor_example_amb_complex_controllers()
    >>> rotor_amb.ndof
    318
    """
    ## Shaft material creation ##
    steel = Material(name="steel", rho=7850, E=1.9e11, Poisson=0.30)
    steel_m12 = Material(name="steel", rho=7700, E=2e11, Poisson=0.31, color="red")

    ## Shaft elements ##
    Li = [
        0.0,
        0.012,
        0.032,
        0.052,
        0.072,
        0.092,
        0.112,
        0.1208,
        0.12724,
        0.13475,
        0.14049,
        0.14689,
        0.15299,
        0.159170,
        0.16535,
        0.180350,
        0.1905,
        0.2063,
        0.2221,
        0.2379,
        0.2537,
        0.2695,
        0.2853,
        0.3011,
        0.3169,
        0.3243,
        0.3363,
        0.358,
        0.364,
        0.3705,
        0.3825,
        0.3986,
        0.4147,
        0.4308,
        0.4469,
        0.4630,
        0.4791,
        0.4952,
        0.5113,
        0.5274,
        0.5356,
        0.5457,
        0.5607,
        0.5669,
        0.5731,
        0.5792,
        0.5856,
        0.5913,
        0.5989,
        0.6053,
        0.6141,
        0.6341,
        0.6461,
    ]
    # Shaft discretization - node positions ## CG: 0.3243
    Li = [round(i, 4) for i in Li]  # Rounding decimal places
    L = [Li[i + 1] - Li[i] for i in range(len(Li) - 1)]  # Element size (e = n-1)
    i_d = [0.0 for _ in L]  # Internal diameter
    o_d1 = [0.0 for _ in L]  # External diameter

    # Adjustments for external diameter
    o_d1[0] = 6.35
    o_d1[1:5] = [32 for _ in range(4)]
    o_d1[5:14] = [34.8 for _ in range(9)]
    o_d1[14:16] = [49.9 for _ in range(2)]
    o_d1[16:27] = [19.05 for _ in range(11)]
    o_d1[27:29] = [54 for _ in range(2)]
    o_d1[29:40] = [19.05 for _ in range(12)]
    o_d1[40:42] = [49.9 for _ in range(2)]
    o_d1[42:51] = [34.8 for _ in range(9)]
    o_d1[51] = 6.35
    o_d = [i * 1e-3 for i in o_d1]  # Conversion to meters

    shaft_elements = [
        ShaftElement(
            L=l,
            idl=idl,
            odl=odl,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
            alpha=2.5,
            beta=1e-4,
        )
        for l, idl, odl in zip(L, i_d, o_d)
    ]

    # Disk elements ##
    n_list = [27, 28, 29]  # Central disk positioning
    n_list_2 = [
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
    ]  # Disks representing lamination
    width = [0.004, 0.007, 0.014]  # Central disk width

    width_2 = [
        0.0088,
        0.0064,
        0.0075,
        0.0057,
        0.0064,
        0.0061,
        0.0062,
        0.0062,
        0.0062,
        0.0062,
        0.0061,
        0.0064,
        0.0057,
        0.0075,
        0.0064,
        0.0088,
    ]  # Lamination disk width

    i_disc_1 = [0.054, 0.054, 0.054]  # Central disk internal diameter
    i_disc_2 = [0.0348] * 16  # Lamination disk internal diameter
    o_disc = [0.1200] * 3  # Central disk external diameter
    o_disc_2 = [0.0498] * 16  # Lamination disk external diameter

    disk_elements_1 = [
        DiskElement.from_geometry(n=n, material=steel, width=m, i_d=Id, o_d=Od)
        for n, m, Id, Od in zip(n_list, width, i_disc_1, o_disc)
    ]

    disk_elements_2 = [
        DiskElement.from_geometry(n=n, material=steel_m12, width=m, i_d=Id, o_d=Od)
        for n, m, Id, Od in zip(n_list_2, width_2, i_disc_2, o_disc_2)
    ]

    disk_elements = [*disk_elements_1, *disk_elements_2]

    # Bearing elements:
    n_list = [12, 43]
    n = 138
    A = 470.3e-6
    i0 = 1.0
    s0 = 0.432e-3
    alpha = 0.392
    c_13 = (
        0.0062
        * (s + 46)
        / s
        * 0.0062
        * (400 / 77)
        * ((s + 77 * 2 * np.pi) / (s + 400 * 2 * np.pi))
        * (409 / 124)
        * ((s + 124 * 2 * np.pi) / (s + 409 * 2 * np.pi))
        * ct.tf(
            [1, 371.964570185032, 5404595.37003653],
            [1, 1301.87599564761, 3095107.94018162],
        )
        * ct.tf(
            [1, 282.743338823081, 22206609.9024511],
            [1, 1337.06183336782, 17458343.225087],
        )
        * 0.4
    )

    c_24 = (
        0.0046
        * 0.0046
        * (s + 35)
        / s
        * (75 / 25)
        * ((s + 25 * 2 * np.pi) / (s + 75 * 2 * np.pi))
        * (1690 / 260)
        * ((s + 260 * 2 * np.pi) / (s + 1690 * 2 * np.pi))
        * ct.tf(
            [1, 1468.38040628787, 3206634.46991393],
            [1, 931.168062524015, 3206634.46991393],
        )
        * ct.tf(
            [1, 1357.16802635079, 6316546.81669719],
            [1, 1240.30077963725, 8720782.44880256],
        )
        * 2
    )
    k_amp = 1.0
    k_sense = 1.0
    bearing_elements = [
        MagneticBearingElement(
            n=n_list[0],
            g0=s0,
            i0=i0,
            ag=A,
            nw=n,
            alpha=alpha,
            controller_transfer_function=c_13,
            k_amp=k_amp,
            k_sense=k_sense,
        ),
        MagneticBearingElement(
            n=n_list[1],
            g0=s0,
            i0=i0,
            ag=A,
            nw=n,
            alpha=alpha,
            controller_transfer_function=c_24,
            k_amp=k_amp,
            k_sense=k_sense,
        ),
    ]

    ## Rotor assembly - 6dof ##
    # Rotor construction:
    if ambs:
        rotor = Rotor(
            shaft_elements=shaft_elements,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
        )
    else:
        rotor = Rotor(
            shaft_elements=shaft_elements,
            disk_elements=disk_elements,
        )

    return rotor


def rotor_example_amb_general_controllers(controller_transfer_function=None):
    r"""This function creates the model of a test rig rotor supported by magnetic bearings.
    Details of the model can be found at doi.org/10.14393/ufu.di.2015.186.

    Returns
    -------
    Rotor object.
    """

    from ross.materials import Material

    steel_amb = Material(name="steel", rho=7850, E=1.9e11, Poisson=0.30)
    steel_m12_amb = Material(name="steel", rho=7600, E=2e11, Poisson=0.31, color="red")

    # Shaft elements:
    # fmt: off
    Li = [
        0.0, 0.012, 0.032, 0.052, 0.072, 0.092, 0.112, 0.1208,
        0.12724, 0.13475, 0.14049, 0.14689, 0.15299, 0.159170,
        0.16535, 0.180350, 0.1905, 0.2063, 0.2221, 0.2379, 0.2537,
        0.2695, 0.2853, 0.3011, 0.3169, 0.3243, 0.3363, 0.358,
        0.364, 0.3705, 0.3825, 0.3986, 0.4147, 0.4308, 0.4469,
        0.4630, 0.4791, 0.4952, 0.5113, 0.5274, 0.5356, 0.5457,
        0.5607, 0.5669, 0.5731, 0.5792, 0.5856, 0.5913, 0.5989,
        0.6053, 0.6141, 0.6341, 0.6461,
    ]

    Li = [round(i, 4) for i in Li]
    L = [Li[i + 1] - Li[i] for i in range(len(Li) - 1)]

    i_d = [0.0 for _ in L]
    o_d1 = [0.0 for _ in L]
    o_d1[0] = 6.35
    o_d1[1:5] = [32 for _ in range(4)]
    o_d1[5:14] = [34.8 for _ in range(9)]
    o_d1[14:16] = [49.9 for _ in range(2)]
    o_d1[16:27] = [19.05 for _ in range(11)]
    o_d1[27:29] = [54 for _ in range(2)]
    o_d1[29:40] = [19.05 for _ in range(12)]
    o_d1[40:42] = [49.9 for _ in range(2)]
    o_d1[42:51] = [34.8 for _ in range(9)]
    o_d1[51] = 6.35
    o_d = [i * 1e-3 for i in o_d1]

    shaft_elements = [
        ShaftElement(
            L=l,
            idl=idl,
            odl=odl,
            material=steel_amb,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
            alpha=2.5,
        )
        for l, idl, odl in zip(L, i_d, o_d)
    ]

    # Disk elements:
    n_list = [27, 28, 29]
    n_list_2 = [6, 7, 8, 9, 10, 11, 12, 13, 43, 44, 45, 46, 47, 48, 49, 50]
    width = [0.004, 0.007, 0.014]
    width_2 = [
        0.0088, 0.0064, 0.0075, 0.0057, 0.0064, 0.0061,
        0.0062, 0.0062, 0.0062, 0.0062,0.0061, 0.0064,
        0.0057, 0.0075, 0.0064, 0.0088,
    ]
    i_disc_1 = [0.054, 0.054, 0.054]
    i_disc_2 = [0.0348] * 16
    o_disc = [0.1200] * 3
    o_disc_2 = [0.0498] * 16
    # fmt: on
    disk_elements_1 = [
        DiskElement.from_geometry(n=n, material=steel_amb, width=m, i_d=Id, o_d=Od)
        for n, m, Id, Od in zip(n_list, width, i_disc_1, o_disc)
    ]

    disk_elements_2 = [
        DiskElement.from_geometry(n=n, material=steel_m12_amb, width=m, i_d=Id, o_d=Od)
        for n, m, Id, Od in zip(n_list_2, width_2, i_disc_2, o_disc_2)
    ]

    disk_elements = [*disk_elements_1, *disk_elements_2]

    # Bearing elements:
    n_list = [12, 43]
    n = 200
    A = 1e-4
    i0 = 1.0
    s0 = 1e-3
    alpha = 0.392
    k_amp = 1.0
    k_sense = 1.0

    if controller_transfer_function is None:
        Kp = 1000
        Ki = 0
        Kd = 5

        bearing_elements = [
            MagneticBearingElement(
                n=n_list[0],
                g0=s0,
                i0=i0,
                ag=A,
                nw=n,
                alpha=alpha,
                k_amp=k_amp,
                k_sense=k_sense,
                kp_pid=Kp,
                kd_pid=Kd,
                ki_pid=Ki,
                tag="Magnetic Bearing 0",
            ),
            MagneticBearingElement(
                n=n_list[1],
                g0=s0,
                i0=i0,
                ag=A,
                nw=n,
                alpha=alpha,
                k_amp=k_amp,
                k_sense=k_sense,
                kp_pid=Kp,
                kd_pid=Kd,
                ki_pid=Ki,
                tag="Magnetic Bearing 1",
            ),
        ]

    else:
        bearing_elements = [
            MagneticBearingElement(
                n=n_list[0],
                g0=s0,
                i0=i0,
                ag=A,
                nw=n,
                alpha=alpha,
                k_amp=k_amp,
                k_sense=k_sense,
                controller_transfer_function=controller_transfer_function,
                tag="Magnetic Bearing 0",
            ),
            MagneticBearingElement(
                n=n_list[1],
                g0=s0,
                i0=i0,
                ag=A,
                nw=n,
                alpha=alpha,
                k_amp=k_amp,
                k_sense=k_sense,
                controller_transfer_function=controller_transfer_function,
                tag="Magnetic Bearing 1",
            ),
        ]

    return Rotor(shaft_elements, disk_elements, bearing_elements)
