import numpy as np
import matplotlib.pyplot as plt
import cmath
from loguru import logger
import control as ct
import time
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ross import (
    Material,
    ShaftElement,
    DiskElement,
    MagneticBearingElement,
    BearingElement,
    Rotor,
)
from ross.plotly_theme import tableau_colors

max_freq = 100  # Hz
x_label = "w [rad/s]"


def get_dof(node, local_dof):
    local_dof_dict = {"x": 0, "y": 1, "alpha": 2, "beta": 3}
    return node * 4 + local_dof_dict[local_dof]


def build_rotor(show_rotor=False):
    logger.info("Initiating rotor build.")

    # --- SHAFT MATERIAL DEFINITION ---
    steel = Material(name="Steel", rho=7850, E=211e9, G_s=81.2e9)

    # --- SHAFT ELEMENTS DEFINITION ---
    L = 0.1  # Length of each element (m)
    i_d = 0  # Internal Diameter (m) - assuming solid shaft
    o_d = 0.05  # External Diameter (m)

    shaft_elements = [
        ShaftElement(
            L=L,
            idl=i_d,
            odl=o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for _ in range(10)
    ]

    # --- DISC DEFINITION ---
    disk = DiskElement.from_geometry(
        n=5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    # --- MAGNETIC BEARING DEFINITION ---
    # Electromagnetic parameters (see ROSS documentation for details)
    g0 = 1e-3  # Air gap
    i0 = 1.0  # Bias current
    ag = 1e-4  # Pole area
    nw = 200  # Winding turns
    alpha = 0.392  # Half of the angle between two poles

    # PID gains
    kp_pid = 1000000  # Kp gain
    kd_pid = 1000000  # Kd gain
    k_amp = 1.0  # Power amplifier gain
    k_sense = 1.0  # Sensor gain

    # Magnetic bearing at node 2
    bearing1 = MagneticBearingElement(
        n=2,
        g0=g0,
        i0=i0,
        ag=ag,
        nw=nw,
        alpha=alpha,
        kp_pid=kp_pid,
        ki_pid=0,
        kd_pid=kd_pid,
        k_amp=k_amp,
        k_sense=k_sense,
    )

    # Simple support at node 8
    kxx = 1e6  # Stiffness in x direction (N/m)
    kyy = 1e6  # Stiffness in y direction (N/m)
    cxx = 1e3  # Damping in x direction (N*s/m)

    bearing2 = BearingElement(n=8, kxx=kxx, kyy=kyy, cxx=cxx)

    # --- ROTOR ASSEMBLY AND VISUALIZATION---
    rotor = Rotor(
        shaft_elements=shaft_elements,
        disk_elements=[disk],
        bearing_elements=[bearing1, bearing2],
    )
    if show_rotor:
        rotor.plot_rotor().show()

    logger.info("Rotor build completed successfully.")
    return rotor


def build_rotor_without_ambs(show_rotor=False):
    logger.info("Initiating rotor build.")

    # --- SHAFT MATERIAL DEFINITION ---
    steel = Material(name="Steel", rho=7850, E=211e9, G_s=81.2e9)

    # --- SHAFT ELEMENTS DEFINITION ---
    L = 0.1  # Length of each element (m)
    i_d = 0  # Internal Diameter (m) - assuming solid shaft
    o_d = 0.05  # External Diameter (m)

    shaft_elements = [
        ShaftElement(
            L=L,
            idl=i_d,
            odl=o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for _ in range(10)
    ]

    # --- DISC DEFINITION ---
    disk = DiskElement.from_geometry(
        n=5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    # Simple support at node 2
    kxx = 1e6  # Stiffness in x direction (N/m)
    kyy = 1e6  # Stiffness in y direction (N/m)
    cxx = 1e3  # Damping in x direction (N*s/m)

    bearing1 = BearingElement(n=2, kxx=kxx, kyy=kyy, cxx=cxx)

    # Simple support at node 8
    kxx = 1e6  # Stiffness in x direction (N/m)
    kyy = 1e6  # Stiffness in y direction (N/m)
    cxx = 1e3  # Damping in x direction (N*s/m)

    bearing2 = BearingElement(n=8, kxx=kxx, kyy=kyy, cxx=cxx)

    # --- ROTOR ASSEMBLY AND VISUALIZATION---
    rotor = Rotor(
        shaft_elements=shaft_elements,
        disk_elements=[disk],
        bearing_elements=[bearing1, bearing2],
    )
    if show_rotor:
        rotor.plot_rotor().show()

    logger.info("Rotor build completed successfully.")
    return rotor


def build_rotor_only_ambs(show_rotor=False):
    logger.info("Initiating rotor build.")

    # --- SHAFT MATERIAL DEFINITION ---
    steel = Material(name="Steel", rho=7850, E=211e9, G_s=81.2e9)

    # --- SHAFT ELEMENTS DEFINITION ---
    L = 0.1  # Length of each element (m)
    i_d = 0  # Internal Diameter (m) - assuming solid shaft
    o_d = 0.05  # External Diameter (m)

    shaft_elements = [
        ShaftElement(
            L=L,
            idl=i_d,
            odl=o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for _ in range(10)
    ]

    # --- DISC DEFINITION ---
    disk = DiskElement.from_geometry(
        n=5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    # --- MAGNETIC BEARING DEFINITION ---
    # Electromagnetic parameters (see ROSS documentation for details)
    g0 = 1e-3  # Air gap
    i0 = 1.0  # Bias current
    ag = 1e-4  # Pole area
    nw = 200  # Winding turns
    alpha = 0.392  # Half of the angle between two poles

    # PID gains
    kp_pid = 1000000  # Kp gain
    kd_pid = 1000000  # Kd gain
    k_amp = 1.0  # Power amplifier gain
    k_sense = 1.0  # Sensor gain

    # Magnetic bearing at node 2
    bearing1 = MagneticBearingElement(
        n=2,
        g0=g0,
        i0=i0,
        ag=ag,
        nw=nw,
        alpha=alpha,
        kp_pid=kp_pid,
        ki_pid=0,
        kd_pid=kd_pid,
        k_amp=k_amp,
        k_sense=k_sense,
    )

    # Magnetic bearing at node 8
    bearing2 = MagneticBearingElement(
        n=8,
        g0=g0,
        i0=i0,
        ag=ag,
        nw=nw,
        alpha=alpha,
        kp_pid=kp_pid,
        ki_pid=0,
        kd_pid=kd_pid,
        k_amp=k_amp,
        k_sense=k_sense,
    )

    # --- ROTOR ASSEMBLY AND VISUALIZATION---
    rotor = Rotor(
        shaft_elements=shaft_elements,
        disk_elements=[disk],
        bearing_elements=[bearing1, bearing2],
    )
    if show_rotor:
        rotor.plot_rotor().show()

    logger.info("Rotor build completed successfully.")
    return rotor


def compute_freq_resp(rotor):
    logger.info("Initiating frequency response computation.")

    speed_range = np.linspace(0, max_freq * 2 * np.pi, 2 * max_freq)
    compute_sensitivite_at = {
        "Bearing 0": {"inp": 9, "out": 9},
        "Bearing 1": {"inp": 33, "out": 33},
    }

    freq_resp = rotor.run_freq_response(
        speed_range=speed_range, compute_sensitivite_at=compute_sensitivite_at
    )

    freq_resp.plot_sensitivity().show()
    np.save(
        "freq_resp",
        freq_resp.freq_resp,
    )
    np.save(
        "speed_range",
        np.array(freq_resp.speed_range),
    )

    logger.info("Frequency response computation completed successfully.")


def get_freq_response_at_mma():
    logger.info("Collecting frequency response at AMB.")

    freq_response = np.load("freq_resp.npy")
    speed_range = np.load("speed_range.npy")

    left_bearing_node = 2
    disk_node = 5

    freq_response_at_mma = freq_response[
        get_dof(disk_node, "y"), get_dof(left_bearing_node, "y"), :
    ]

    mag_W = [abs(z) for z in freq_response_at_mma]
    phase_W = [cmath.phase(z) for z in freq_response_at_mma]

    logger.info("Frequency response at AMB collected successfully.")
    return mag_W, phase_W, speed_range


def plot_freq_resp():
    logger.info("Initiating frequency response plot.")

    mag_W, phase_W, speed_range = get_freq_response_at_mma()

    # Displacement
    # Magnitude
    plt.figure(figsize=(8, 6), dpi=130)
    plt.subplot(2, 1, 1)
    plt.plot(speed_range, mag_W)
    plt.legend(["$\mathrm{G_w(s)}$"])
    plt.xlabel(x_label)
    plt.ylabel("Magnitude [m/N]")
    plt.title("Frequency Response")
    plt.yscale("log")
    plt.xlim([np.min(speed_range), np.max(speed_range)])
    plt.grid()
    plt.tight_layout()

    # Fase
    plt.subplot(2, 1, 2)
    plt.plot(speed_range, phase_W)
    plt.legend(["$\mathrm{G_w(s)}$"])
    plt.xlabel(x_label)
    plt.ylabel("Phase [rad]")
    plt.xlim([np.min(speed_range), np.max(speed_range)])
    plt.grid()
    plt.tight_layout()

    logger.info("Frequency response plot completed successfully.")


def compute_sensitivity_call():
    logger.info("Initiating sensitivity computation.")
    mag_W, phase_W, speed_range = get_freq_response_at_mma()
    start_time = time.time()
    mag_S, phase_S, speed_range = compute_sensitivity(
        kp_pid=1000000,
        ki_pid=0,
        kd_pid=1000000,
        mag_W=mag_W,
        phase_W=phase_W,
        speed_range=speed_range,
    )
    end_time = time.time()
    logger.info(f"Sensitivity computation time: {end_time - start_time} seconds.")
    return mag_S, phase_S, speed_range


def compute_sensitivity(kp_pid, ki_pid, kd_pid, mag_W, phase_W, speed_range):
    # Controller frequency response computation
    s = ct.tf("s")
    C = kp_pid + ki_pid / s + kd_pid * s
    mag_C, phase_C, _ = ct.frequency_response(C, speed_range)

    # Close-loop frequency response computation
    mag_T = mag_W * mag_C
    phase_T = phase_W + phase_C

    complex_T = [
        complex(
            mag_phase[0] * np.cos(mag_phase[1]), mag_phase[0] * np.sin(mag_phase[1])
        )
        for mag_phase in zip(mag_T, phase_T)
    ]

    # Sensitivity computation
    complex_S = [1 - z for z in complex_T]

    mag_S = [abs(z) for z in complex_S]
    phase_S = [cmath.phase(z) for z in complex_S]

    return mag_S, phase_S, speed_range


def plot_sensitivity(mag_S, phase_S, speed_range):
    logger.info("Initiating sensitivity response plot.")

    # Displacement
    # Magnitude
    plt.figure(figsize=(8, 6), dpi=130)
    plt.subplot(2, 1, 1)
    plt.plot(speed_range, mag_S)
    plt.legend(["S(s)"])
    plt.xlabel(x_label)
    plt.ylabel("Magnitude [m/N]")
    plt.title("Frequency Response")
    plt.yscale("log")
    plt.xlim([np.min(speed_range), np.max(speed_range)])
    plt.grid()
    plt.tight_layout()

    # Fase
    plt.subplot(2, 1, 2)
    plt.plot(speed_range, phase_S)
    plt.legend(["S(s)"])
    plt.xlabel(x_label)
    plt.ylabel("Phase [rad]")
    plt.xlim([np.min(speed_range), np.max(speed_range)])
    plt.grid()
    plt.tight_layout()

    logger.info("sensitivity plot completed successfully.")


def plot_sensitivity_plotly(mag_S, phase_S, speed_range):
    frequency_units = "rad/s"
    amplitude_units = "m/N"

    logger.info("Initiating sensitivity response plot.")

    fig = make_subplots(rows=2, cols=1)

    # Magnitude
    fig.add_trace(
        go.Scatter(
            x=speed_range,
            y=mag_S,
            mode="lines",
            line=dict(color=list(tableau_colors)[0]),
            name=f"inp: node 0 | dof: 0<br>out: node 0 | dof: 0",
            legendgroup=f"inp: node 0 | dof: 0<br>out: node 0 | dof: 0",
            showlegend=True,
            hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=f"Frequency ({frequency_units})",
        range=[np.min(speed_range), np.max(speed_range)],
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text=f" Magnitude ({amplitude_units})", row=1, col=1)

    # Phase
    fig.add_trace(
        go.Scatter(
            x=speed_range,
            y=phase_S,
            mode="lines",
            line=dict(color=list(tableau_colors)[0]),
            showlegend=False,
            hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text=f"Frequency ({frequency_units})",
        range=[np.min(speed_range), np.max(speed_range)],
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=f" Magnitude ({amplitude_units})", row=2, col=1)

    logger.info("sensitivity plot completed successfully.")

    return fig


def main():
    start_time = time.time()
    # rotor = build_rotor(show_rotor=False)
    rotor = build_rotor_without_ambs(show_rotor=False)
    # rotor = build_rotor_only_ambs(show_rotor=False)
    compute_freq_resp(rotor)
    # plot_freq_resp()
    # mag_S, phase_S, speed_range = compute_sensitivity_call()
    # plot_sensitivity(mag_S, phase_S, speed_range)
    # fig = plot_sensitivity_plotly(mag_S, phase_S, speed_range)
    # plt.show()
    # fig.show()

    logger.info("Execution completed successfully.")
    end_time = time.time()

    logger.info(f"Execution time: {end_time - start_time}.")


if __name__ == "__main__":
    main()
