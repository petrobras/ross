# %%
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# %%
import sys

sys.path.append("/home/jguarato/Documents/ross")

import ross as rs
import numpy as np

from prettytable import PrettyTable
from scipy.signal import find_peaks

import plotly.graph_objects as go
import plotly.io as pio

from ross.units import Q_
from ross.probe import Probe
from ross.utils import compute_dfft

# %% [markdown]
# # Electrical modeling

# %%
# Load          100%  75%   50%   25%
# Power factor  0.87  0.84  0.7   0.5
# Efficiency    0.971 0.97  0.963 0.94
efficiency = 0.971
power_factor = 0.87

frequency = Q_(60, "Hz")
current_rated = 422.7
T_rated = 13192
P_rated = 2474e3
V_rated = 4000

# S_base = P_rated / (efficiency * power_factor)
S_base = np.sqrt(3.0) * V_rated * current_rated
Z_base = (V_rated**2) / S_base

# PARAMETERS FOR EQUIVALENT MODEL (pu)         sn    s=1
# STATOR RESISTANCE (r1)                       0.005 0.005
# STATOR REACTANCE (X1)                        0.131 0.105
# ROTOR RESISTANCE REGARD. TO THE STATOR (r2)  0.005 0.02
# ROTOR REACTANCE REGARD. TO THE STATOR (X2)   0.105 0.064
# MAGNETIZATION RESISTANCE (rc) (IRON LOSSES)  199.5 188.9
# MAGNETIZATION REACTANCE (Xm)                 3.363 3.282
r1_pu = 0.005  # Resistência do estator em p.u.
x1_pu = 0.131  # Reatância de dispersão do estator em p.u.
r2_pu = 0.005  # Resistência do rotor em p.u.
x2_pu = 0.105  # Reatância de dispersão do rotor em p.u.
xm_pu = 3.363  # Reatância de magnetização em p.u.

electric_params = dict(
    n_poles=4,
    # Operação
    power_rated=P_rated,
    speed_rated=Q_(1791, "RPM"),
    frequency_rated=frequency,
    voltage_rated=V_rated,
    frequency_net=frequency,
    voltage_net=V_rated,
    # Conversão de p.u. para Ohms
    stator_resistance=r1_pu * Z_base,
    rotor_resistance=r2_pu * Z_base,
    stator_reactance=x1_pu * Z_base,
    rotor_reactance=x2_pu * Z_base,
    mutual_reactance=xm_pu * Z_base,
    # Parâmetros da carga mecânica
    Ip_load=Q_(104.7022064294543, "kg*m**2"),  # Q_(140, "kg*m**2"),
    viscosity_coeff=0.0,
    # ROSS
    tag="M-C-5283001",
    scale_factor=0.65,
)

# %% [markdown]
# # FEM model

# %%
diretorio = "/home/jguarato/Documents/ross-files/motor/tomls"
# motor = rs.Rotor.load(f"{diretorio}/motor_replan2.toml")
motor = rs.Rotor.load(f"{diretorio}/motor_replan.toml")
coupling_ls = rs.Rotor.load(f"{diretorio}/coupling_ls.toml")
rotor_ls_gearbox = rs.Rotor.load(f"{diretorio}/rotor_ls_gearbox.toml")
rotor_hs_gearbox = rs.Rotor.load(f"{diretorio}/rotor_hs_gearbox.toml")
coupling_hs = rs.Rotor.load(f"{diretorio}/coupling_hs.toml")
compressor = rs.Rotor.load(f"{diretorio}/compressor.toml")

# %%
load = rs.Rotor.concatenate(coupling_ls, rotor_ls_gearbox)

print(f"Motor Ip = {motor.Ip}\nMotor mass = {motor.m}")
print(f"Load Ip = {load.Ip}\nLoad mass = {load.m}")

# %%
electric_motor = rs.ElectricMotor(
    n=0,
    shaft_elements=motor.shaft_elements,
    bearing_elements=motor.bearing_elements,
    disk_elements=motor.disk_elements,
    point_mass_elements=motor.point_mass_elements,
    **electric_params,
)

# %%
rotor_ls = rs.Rotor.concatenate(coupling_ls, rotor_ls_gearbox)
rotor_hs = rs.Rotor.concatenate(rotor_hs_gearbox, coupling_hs, compressor)

n_gear_ls = next(e.n for e in rotor_ls.disk_elements if isinstance(e, rs.GearElement))
n_gear_hs = next(e.n for e in rotor_hs.disk_elements if isinstance(e, rs.GearElement))
coupled_nodes = (n_gear_ls, n_gear_hs)

multirotor = rs.MultiRotor(
    rotor_ls,
    rotor_hs,
    coupled_nodes=coupled_nodes,
    gear_mesh_stiffness=1.38e8,
    orientation_angle=0,
    position="below",
)

# %% [markdown]
# # Final model

# %%
rotor = multirotor.add_motor(electric_motor)

rotor.plot_rotor().show()

# %% [markdown]
# # Results

# %% [markdown]
# ## Modal analysis

# %%
# omega = speed[-1]
# modal = rotor.run_modal(speed=omega, num_modes=60)

# table2 = modal.format_table("Hz")
# table2.add_column("Type", [sh.mode_type for sh in modal.shapes])

# print(f"Speed = {Q_(omega, "rad/s").to("Hz").m} Hz | {Q_(omega, "rad/s").to("RPM").m} RPM")
# # print(table2.get_string(row_filter=lambda row: row[5] == "Torsional"))
# print(table2)

# %% [markdown]
# ## Running time response

# %%
t0 = 0
tf = 16
dt = 1e-4
t = np.linspace(t0, tf, int((tf - t0) / dt) + 1)

node = [13]
unbalance_magnitude = [2.26643e-2]
unbalance_phase = [0]

results1 = rotor.run_with_motor(
    t,
    node,
    unbalance_magnitude,
    unbalance_phase,
    drive_mode="VFD_VF",
    load_torque_entrance_time=10.0,
    load_torque_ratio=1.0,
    time_ramp=0.6667,
    frequency_ref=frequency,
    steady_state=True,
    time_step=1e-6,
    verbose=True,
    newmark_type="robust",
)

# %% [markdown]
# ### Plot

# %%
nodes = [25, 30, 36, 91, 95, 100]

# %% [markdown]
# #### Auxiliary functions


# %%
def plot_torsional(self, nodes, add_name, fig=None):
    if fig is None:
        fig = go.Figure()

    for n in nodes:
        doft = n * rotor.number_dof + 5
        doft_motor = rotor.motor.n_torque * rotor.number_dof + 5

        theta_z = self.yout[:, doft_motor] - self.yout[:, doft]

        fig.add_trace(
            go.Scatter(x=self.t, y=theta_z, mode="lines", name=f"{add_name} — n={n}")
        )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Relative displacement (rad)")
    fig.update_layout(title="Torsional DOF")

    return fig


def plot_dfft_torsional(self, nodes, add_name, frequency_units="Hz", fig=None):
    if fig is None:
        fig = go.Figure()

    dt = np.diff(self.t).min()

    for n in nodes:
        doft = n * rotor.number_dof + 5
        theta_z = self.yout[:, doft]
        freq, amp, _ = compute_dfft(theta_z, dt)

        freq_ = rs.Q_(freq, "Hz").to(frequency_units).m

        fig.add_trace(
            go.Scatter(
                x=freq_,
                y=amp,
                mode="lines",
                name=f"{add_name} — n={n}",
                hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude: %{{y:.2e}}",
            )
        )

    fig.update_xaxes(title_text=f"Frequency ({frequency_units})")
    fig.update_yaxes(title_text="Amplitude (rad)")
    fig.update_layout(title="Torsional DOF")

    return fig


# %% [markdown]
# #### Final comparison

# %%
ref_name = "With torque"

fig_x = results1.plot_dfft(
    probe=[rs.Probe(n, Q_(0, "deg"), tag=f"{ref_name} n={n}") for n in nodes]
)
fig_y = results1.plot_dfft(
    probe=[rs.Probe(n, Q_(90, "deg"), tag=f"{ref_name} n={n}") for n in nodes]
)
fig_tz = plot_dfft_torsional(results1, nodes, ref_name)

# %%
titles = ["Lateral DOF (X)", "Lateral DOF (Y)", "Torsional DOF"]
figs = [fig_x, fig_y, fig_tz]
for fig, title in zip(figs, titles):
    fig.update_layout(title=title)
    fig.update_yaxes(type="log")
    # fig.update_xaxes(range=(0, 400))
    fig.show()

# %%
results1.plot_1d(
    probe=[rs.Probe(n, Q_(0, "deg"), tag=f"{ref_name} n={n}") for n in nodes]
).show()
