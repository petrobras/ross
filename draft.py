import numpy as np
import matplotlib.pyplot as plt
from ross.defects import (
    MisalignmentFlexAngular,
    MisalignmentFlexParallel,
    MisalignmentFlexCombined,
)
import ross as rs

import plotly.io as pio

dt = 0.001
t = np.arange(0, 10 + dt, dt)

speedI = 1200
speedF = 1200
lambdat = 0.00001

warI = speedI * np.pi / 30
warF = speedF * np.pi / 30

tI = t[0]
tF = t[-1]

Faxial = 0
TorqueI = 0
TorqueF = 0

sA = (warI * np.exp(-lambdat * tF) - warF * np.exp(-lambdat * tI)) / (
    np.exp(-lambdat * tF) - np.exp(-lambdat * tI)
)
sB = (warF - warI) / (np.exp(-lambdat * tF) - np.exp(-lambdat * tI))

sAT = (TorqueI * np.exp(-lambdat * tF) - TorqueF * np.exp(-lambdat * tI)) / (
    np.exp(-lambdat * tF) - np.exp(-lambdat * tI)
)
sBT = (TorqueF - TorqueI) / (np.exp(-lambdat * tF) - np.exp(-lambdat * tI))

SpeedV = sA + sB * np.exp(-lambdat * t)
TorqueV = sAT + sBT * np.exp(-lambdat * t)
AccelV = -lambdat * sB * np.exp(-lambdat * t)

TetaV = sA * t - (sB / lambdat) * np.exp(-lambdat * t) + (sB / lambdat)
# TetaV = np.loadtxt("data/angular_position.txt")

Radius = (1 / 2) * 19 * 1 * 10 ** (-3)
coup = 1  # posicao do acoplamento - para correcao na matriz de rigidez
kCOUP = 5e5  # k3 - rigidez no acoplamento
nodeI = 1  # no inicial do acoplamento
nodeF = 2  # no final do acoplamento
eCOUPx = 2 * 10 ** (-4)  # Distancia de desalinhamento entre os eixos - direcao x
eCOUPy = 2 * 10 ** (-4)  # Distancia de desalinhamento entre os eixos - direcao z
kd = 40 * 10 ** (3)  # Rigidez radial do acoplamento flexivel
ks = 38 * 10 ** (3)  # Rigidez de flexão do acoplamento flexivel
alpha = 5 * np.pi / 180  # Angulo do desalinhamento angular (rad)
fib = np.arctan2(eCOUPy, eCOUPx)  # Angulo de rotacao em torno de y;
TD = 0  # Torque antes do acoplamento
TL = 0  # Torque dopois do acoplamento
Nele = 0

# teste1 = MisalignmentFlexParallel(
#     TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL, n1=0, n2=1
# )
# teste2 = MisalignmentFlexAngular(TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL)

misalignment = MisalignmentFlexCombined(
    TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL, n1=0, n2=1
)

steel = rs.materials.steel
steel.rho = 7.85e3
steel.E = 2.17e11
#  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
i_d = 0
o_d = 0.019
n = 33
# fmt: off
L = np.array(
        [0,25,64,104,124,143,175,207,239,271,303,335,345,355,380,408,436,466,496,526,556,586,614,647,657,667,702,737,772,807,842,862,881,914]
    )/ 1000

# fmt: on

L = [L[i] - L[i - 1] for i in range(1, len(L))]
shaft_elem = [
    rs.ShaftElement6DoF(
        material=steel,
        L=l,
        idl=i_d,
        odl=o_d,
        idr=i_d,
        odr=o_d,
        alpha=2.0501,
        beta=1.4e-8,
        rotary_inertia=True,
        shear_effects=True,
    )
    for l in L
]
Id = 0.003844540885417
Ip = 2 * Id

disk0 = rs.DiskElement6DoF(n=12, m=2.6375, Id=Id, Ip=Ip)
disk1 = rs.DiskElement6DoF(n=24, m=2.6375, Id=Id, Ip=Ip)

kxx1 = 6.7072e5
kyy1 = 7.8114e5
kzz = 0
cxx1 = 10.4
cyy1 = 7.505
czz = 0
kxx2 = 2.010e6
kyy2 = 1.1235e8
cxx2 = 13.4
cyy2 = 8.4553
bearing0 = rs.BearingElement6DoF(
    n=4, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz
)
bearing1 = rs.BearingElement6DoF(
    n=31, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz
)

rotor = rs.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

# rotor6.transfer_matrix(speed=1500)
# camp6 = rotor6.run_campbell(np.linspace(0, 400, 101), frequencies=18)

# # plotting Campbell Diagram
# fig = camp6.plot()
# pio.show(fig)

print("")


# rotor = rotor_example()
node = 14
probe1 = (14, 0)
probe2 = (22, 0)
# t = np.linspace(0, 20, size)
F = np.zeros((len(t), rotor.ndof))
# F[:, 6 * node] = 10 * np.cos(2 * t)
# F[:, 6 * node + 1] = 10 * np.sin(2 * t)
# response = rotor.run_time_response(speedI * np.pi / 30, F, t, defect=None)
response = rotor.run_unbalance_response(
    [12, 24],
    [100e-06, 130e-06],
    [-np.pi / 2, -np.pi / 2],
    frequency=np.arange(0, speedI * np.pi / 30, 0.1),
)

# >>> response = rotor.run_unbalance_response(node=3,
# ...                                         unbalance_magnitude=10.0,
# ...                                         unbalance_phase=0.0,
# ...                                         frequency=speed)
# response.yout[:, 77]  # doctest: +ELLIPSIS

# plot time response for a given probe:
fig1 = response.plot(probe=[probe1, probe2])

# plot orbit response - plotting 2D nodal orbit:

# fig2 = response.plot_2d(node=node)

# plot orbit response - plotting 3D orbits - full rotor model:
# fig3 = response.plot_3d()
pio.show(fig1)
# pio.show(fig3)
# pio.show(fig2)

print("")
