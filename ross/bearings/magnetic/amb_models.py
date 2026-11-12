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


def _amb_test_rig_rotor(bearing_elements=None):
    """Rotor of the LMEst/UFU magnetic bearing test rig, without bearings.

    The shaft and disk come from the consolidated model R6 of the test rig
    (LMEst/UFU, unpublished, 2026; the reference will be added after
    publication). R6 was fitted to free-free impact tests of the rotor:

    - 645 mm shaft, 5.89 kg in total, 20 mm maximum element length (46 nodes).
      The mesh converges to within 0.004 % of the 5 mm mesh of R6 in the first
      three free-free bending frequencies.
    - The laminated sleeves under the actuators are concentric shaft elements
      on the same nodes as the core, so they add mass and stiffness.
    - Each region has its own equivalent E and rho from the R6 fit. The four
      short elements with reduced E are the local flexibility of the shaft
      steps, which keeps the geometric lengths. They are not real materials.
    - In R6 the center of mass of the disk is 1.03 mm away from its node. A
      standard DiskElement cannot represent this offset, so the disk sits on
      the node with the R6 mass and inertias (referred to its center of mass).
      This changes the free-free bending frequencies of R6 (106.91, 400.90 and
      769.28 Hz) by -0.05 %, +0.25 % and -0.35 %.
    - Stiffness-proportional damping, beta = 2.1248e-7 s (alpha = 0), fitted to
      the effective damping ratios of the first three free-free bending modes
      identified with R6 (about 1e-4). Mass-proportional damping is left out
      because it would also damp the rigid-body motion.

    Parameters
    ----------
    bearing_elements : list, optional
        Bearing elements of the rotor. Defaults to no bearings (free-free).

    Returns
    -------
    rotor : ross.Rotor
        Rotor of the test rig.
    """
    # fmt: off
    node_positions_mm = [
        0.0, 12.0, 27.0, 42.0, 58.666666667, 75.333333333,
        92.0, 112.0, 127.24, 132.0, 142.495, 152.99,
        165.3, 180.5, 190.5, 207.733333333, 224.966666667, 242.2,
        261.46, 280.72, 299.98, 319.24, 338.5, 348.5,
        360.1, 361.0, 370.1, 373.5, 393.175, 412.85,
        432.525, 452.2, 468.66, 485.12, 501.58, 518.04,
        534.5, 544.5, 559.6, 572.01, 584.885, 597.76,
        613.0, 618.9, 633.0, 645.0,
    ]
    # Equivalent materials of R6: (E (Pa), rho (kg/m^3), Poisson)
    materials = [
        (201912612668.44516, 7264.434381617933, 0.3),    # 0: shaft, drive end
        (200000000000.0, 6700.479950384487, 0.31),       # 1-5: lamination, drive
        (200000000000.0, 6725.803272022205, 0.31),       #      end (axial mass
        (200000000000.0, 6745.118635601375, 0.31),       #      profile of R6)
        (200000000000.0, 6771.6954616601615, 0.31),
        (200000000000.0, 6800.570379157571, 0.31),
        (201912612668.44516, 10173.528920453931, 0.3),   # 6: sleeve, drive end
        (9546884519.119942, 10173.528920453931, 0.3),    # 7: sleeve, step flexibility
        (201912612668.44516, 7878.982875436916, 0.3),    # 8: shaft, non-drive end
        (133955557750.27191, 7878.982875436916, 0.3),    # 9: step flexibility (disk)
        (179687186306.40756, 7878.982875436916, 0.3),    # 10: step flexibility (disk)
        (9546884519.119942, 7878.982875436916, 0.3),     # 11: sleeve, step flexibility
        (200000000000.0, 7256.193338209257, 0.31),       # 12: lamination, non-drive end
    ]
    # (node, outer diameter (mm), inner diameter (mm), material index)
    elements = [
        (0, 6.35, 0.0, 0), (1, 32.0, 0.0, 0), (2, 32.0, 0.0, 0),
        (3, 32.0, 0.0, 0), (4, 32.0, 0.0, 0), (5, 32.0, 0.0, 0),
        (6, 34.8, 0.0, 0),
        (7, 34.8, 0.0, 0), (7, 49.9, 34.8, 1),
        (8, 34.8, 0.0, 0), (8, 49.9, 34.8, 2),
        (9, 34.8, 0.0, 0), (9, 49.9, 34.8, 3),
        (10, 34.8, 0.0, 0), (10, 49.9, 34.8, 4),
        (11, 34.8, 0.0, 0), (11, 49.9, 34.8, 5),
        (12, 49.9, 0.0, 6), (13, 49.9, 0.0, 7),
        (14, 19.05, 0.0, 8), (15, 19.05, 0.0, 8), (16, 19.05, 0.0, 8),
        (17, 19.05, 0.0, 8), (18, 19.05, 0.0, 8), (19, 19.05, 0.0, 8),
        (20, 19.05, 0.0, 8), (21, 19.05, 0.0, 8), (22, 19.05, 0.0, 9),
        (23, 54.0, 0.0, 8),
        (24, 19.05, 0.0, 10), (25, 19.05, 0.0, 10), (26, 19.05, 0.0, 8),
        (27, 19.05, 0.0, 8), (28, 19.05, 0.0, 8), (29, 19.05, 0.0, 8),
        (30, 19.05, 0.0, 8), (31, 19.05, 0.0, 8), (32, 19.05, 0.0, 8),
        (33, 19.05, 0.0, 8), (34, 19.05, 0.0, 8), (35, 19.05, 0.0, 8),
        (36, 49.9, 0.0, 11), (37, 49.9, 0.0, 8),
        (38, 34.8, 0.0, 8), (38, 49.9, 34.8, 12),
        (39, 34.8, 0.0, 8), (39, 49.9, 34.8, 12),
        (40, 34.8, 0.0, 8), (40, 49.9, 34.8, 12),
        (41, 34.8, 0.0, 8), (41, 49.9, 34.8, 12),
        (42, 34.8, 0.0, 8), (43, 34.8, 0.0, 8), (44, 6.35, 0.0, 8),
    ]
    # fmt: on

    # The equivalent materials share two names, so that the rotor plot shows
    # one legend entry for the shaft and one for the laminations. Each shaft
    # element keeps its own E and rho.
    laminations = {1, 2, 3, 4, 5, 12}
    steels = [
        Material(
            name="amb_rig_lamination" if i in laminations else "amb_rig_shaft",
            E=E,
            rho=rho,
            Poisson=poisson,
            color="#468f9c" if i in laminations else "#525252",
        )
        for i, (E, rho, poisson) in enumerate(materials)
    ]
    shaft_elements = [
        ShaftElement(
            n=n,
            L=(node_positions_mm[n + 1] - node_positions_mm[n]) * 1e-3,
            idl=idl * 1e-3,
            odl=odl * 1e-3,
            material=steels[material],
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
            alpha=0.0,
            beta=2.12480562e-7,
        )
        for n, odl, idl, material in elements
    ]

    # Disk on its mounting node (360.1 mm); mass and inertias of R6 at the
    # disk center of mass.
    disk = DiskElement(
        n=24,
        m=1.7658539830837185,
        Id=0.0022225435635540654,
        Ip=0.0037500871621930173,
        tag="Disk",
    )

    return Rotor(
        shaft_elements=shaft_elements,
        disk_elements=[disk],
        bearing_elements=bearing_elements or [],
    )


def rotor_example_amb_complex_controllers(ambs=True):
    """
    Create the LMEst/UFU magnetic bearing test rig with its controllers.

    The rotor is the consolidated model R6 of the test rig (see
    ``_amb_test_rig_rotor``). The magnetic bearings use the actuator of the
    test rig and the controllers of its MBScope software, with collocated
    sensors.

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
    276
    """
    # Bearing elements:
    # Actuator data from the manufacturer's datasheet: each axis is a
    # horseshoe electromagnet with two coils of 138 turns (n = 276 turns per
    # electromagnet) and a pole-face area of 470.3 mm^2 per axis, i.e.
    # A = 235.15 mm^2 per pole, as required by the ki and ks formulas of
    # MagneticBearingElement. i0 = 1 A is the bias current used in the
    # experiments. This ideal model gives
    # ki = 111.5 N/A and ks = -258 kN/m, about 10 % above the values
    # identified on the rig (96-104 N/A and -203 to -235 kN/m), since
    # fringing and iron reluctance are not modeled.
    n_list = [11, 39]  # actuator centers at 152.99 mm and 572.01 mm
    n = 276
    A = 235.15e-6
    i0 = 1.0
    s0 = 0.432e-3
    alpha = 0.392

    # Controllers of the LMEst rig (MBResearch, MBScope). The control filter
    # is specified in A/um, so 1e6 converts the rotor displacement from m to
    # um. Sensor and amplifier gains are cancelled inside the controller
    # software (position sensitivity in um/V, current sensitivity in A/V),
    # hence k_amp = k_sense = 1. The proportional gains are 0.0062 A/um
    # (c_13) and 0.0046 A/um (c_24); levitation requires the loop gain to
    # exceed |ks| / ki = i0 / s0 (about 0.0023 A/um for this actuator).
    c_13 = (
        1e6
        * 0.0062
        * (s + 46)
        / s
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
        1e6
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

    if not ambs:
        return _amb_test_rig_rotor()

    return _amb_test_rig_rotor(bearing_elements)


def rotor_example_amb_general_controllers(controller_transfer_function=None):
    r"""Create the magnetic bearing test rig rotor with a generic actuator and controller.

    The rotor is the consolidated model R6 of the LMEst/UFU test rig (see
    ``_amb_test_rig_rotor``), the same as in
    ``rotor_example_amb_complex_controllers``. The actuator has round,
    illustrative parameters and the controller is a PD controller, or the
    transfer function given by the user. The original controller study of the
    test rig is at doi.org/10.14393/ufu.di.2015.186.

    Parameters
    ----------
    controller_transfer_function : control.TransferFunction, optional
        Controller of both magnetic bearings. Defaults to a PD controller.

    Returns
    -------
    Rotor object.
    """

    # Bearing elements:
    n_list = [11, 39]  # actuator centers at 152.99 mm and 572.01 mm
    n = 200
    A = 1e-4
    i0 = 1.0
    s0 = 1e-3
    alpha = 0.392
    k_amp = 1.0
    k_sense = 1.0

    if controller_transfer_function is None:
        # Levitation requires ki * k_amp * k_sense * Kp > |ks|, i.e.
        # Kp > i0 / s0 = 1000 for this geometry. Kp = 3 * i0 / s0 gives a
        # net static stiffness of 2 * |ks| per axis.
        Kp = 3000
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

    return _amb_test_rig_rotor(bearing_elements)


def rotor_example_amb_simple():
    steel = Material(name="steel", rho=7850, E=1.9e11, Poisson=0.30)

    L = 0.1
    idl = 0.0
    odl = 0.05
    n_elements = 10

    shaft_elements = [
        ShaftElement(
            n=i,
            L=L,
            idl=idl,
            odl=odl,
            material=steel,
            alpha=10,
            beta=1e-4,
        )
        for i in range(n_elements)
    ]

    disk = DiskElement(
        n=int(np.round(n_elements / 2)), m=10, Id=0.01, Ip=0.02, tag="Disk_1"
    )

    amb_params = {
        "g0": 1e-3,
        "i0": 1.0,
        "ag": 1e-4,
        "nw": 200,
        "kp_pid": 1e5,
        "kd_pid": -1e-5,
        "ki_pid": 2e-1,
        "k_amp": 1.0,
        "k_sense": 1.0,
    }

    bearings = [
        MagneticBearingElement(n=n, **amb_params, tag=f"AMB_{n}")
        for n in [1, n_elements - 1]
    ]

    rotor = Rotor(
        shaft_elements=shaft_elements, disk_elements=[disk], bearing_elements=bearings
    )

    return rotor
