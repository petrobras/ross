"""Co-axial rotor example (Friswell et al., Dynamics of Rotating Machines, Fig. 6.45).

Inner rotor: book nodes 1-8  -> ROSS nodes 0-7 (solid shaft, d = 30 mm)
Outer rotor: book nodes 9-13 -> ROSS nodes 8-12 (tube, 50/60 mm)
The outer rotor co-rotates at 1.5 times the inner rotor speed.

Figure 6.45. The example co-axial rotor. The nodes are indicated by n, the bearings by b,
and the disks by d, followed by the number. The nodes on the inner rotor are indicated by a
cross and those on the outer rotor by dots on the outer sections linked by lines.

Table 6.1. Disk properties for the co-axial example in
Disk, Mass (kg), Id (kg m2), Ip (kg m2), Node
d1, 10.5, 0.043, 0.086, 2
d2, 7.0, 0.021, 0.042, 10
d3, 3.5, 0.013, 0.026, 12
d4, 7.0, 0.034, 0.068, 7

The outer rotor spin speed is 1.5 times the spin speed of the inner rotor and it
co-rotates. The system has the following dimensions and properties: The shaft
of the inner rotor is solid with a diameter of 30 mm. The shaft of the outer rotor
has an inside diameter of 50mm and an outside diameter of 60mm. Relative
to the left bearing of the inner rotor, the positions of nodes 1 through 13 are
0, 0.076, 0.159, 0.254, 0.324, 0.406, 0.457, 0.508, 0.152, 0.203, 0.279, 0.356, and
0.406 m, respectively. Disks d l, d2, d3, and d4 are located at nodes 2, 10, 12,
and 7, respectively. Table 6.1 lists the disk masses and inertias. Bearings b1, b2,
and b4 are located at nodes 1, 9, and 8, respectively, and the inter-shaft bear
ing, b3, connects nodes 6 and 13. Table 6.2 lists the bearing properties. The ro
tors are made from steel with the properties E = 207 GPa and p = 8,300 kg/m3.
Plot the Campbell diagram for this system and calculate the response to
unbalance forces of magnitude 0.0001 kgm on each of disks d1 and d2 in
turn.

Table 6.2. Bearing properties for the co-axial example in
Bearing, kxx(MN/m), kyy(MN/m), Node
b1, 26, 52, 1
b2, 18, 36, 9
b3, 9, 9, 6 and 13
b4, 18, 36, 8
"""

import numpy as np

import ross as rs
from ross.results import ForcedResponseResults

Q_ = rs.Q_

SPEED_RATIO = 1.5

steel = rs.Material(name="Steel_Friswell", rho=8300, E=207e9, Poisson=0.3)

inner_pos = np.array([0, 0.076, 0.159, 0.254, 0.324, 0.406, 0.457, 0.508])
outer_pos = np.array([0.152, 0.203, 0.279, 0.356, 0.406])

inner_shaft = [
    rs.ShaftElement(L=L, idl=0, odl=0.030, material=steel) for L in np.diff(inner_pos)
]
outer_shaft = [
    rs.ShaftElement(L=L, idl=0.050, odl=0.060, material=steel)
    for L in np.diff(outer_pos)
]

# Book node -> ROSS node: n_ross = n_book - 1
disks = [
    rs.DiskElement(n=1, m=10.5, Id=0.043, Ip=0.086, tag="d1"),
    rs.DiskElement(n=9, m=7.0, Id=0.021, Ip=0.042, tag="d2"),
    rs.DiskElement(n=11, m=3.5, Id=0.013, Ip=0.026, tag="d3"),
    rs.DiskElement(n=6, m=7.0, Id=0.034, Ip=0.068, tag="d4"),
]

bearings = [
    rs.BearingElement(n=0, kxx=26e6, kyy=52e6, cxx=0, tag="b1"),
    rs.BearingElement(n=8, kxx=18e6, kyy=36e6, cxx=0, tag="b2"),
    rs.BearingElement(n=5, n_link=12, kxx=9e6, kyy=9e6, cxx=0, tag="b3"),
    rs.BearingElement(n=7, kxx=18e6, kyy=36e6, cxx=0, tag="b4"),
]

rotor = rs.CoAxialRotor(
    [inner_shaft, outer_shaft], disks, bearings, speed_ratio=SPEED_RATIO
)
# print("Node positions (m):", np.round(rotor.nodes_pos, 3))
# print(f"Rotor mass: {rotor.m:.2f} kg")
rotor.plot_rotor().show()


# def apply_outer_speed_ratio(rotor, outer_elements, ratio):
#     """Scale the gyroscopic matrix of the outer rotor elements by the speed ratio.

#     The rotor speed passed to the analyses is the inner rotor speed, so the
#     gyroscopic contribution of every element spinning with the outer rotor is
#     multiplied by ``ratio``.
#     """
#     for elm in outer_elements:
#         dofs = list(elm.dof_global_index.values())
#         rotor.G0[np.ix_(dofs, dofs)] += (ratio - 1) * elm.G()


# outer_elements = [sh for sh in rotor.shaft_elements if sh.n >= 8] + [
#     d for d in rotor.disk_elements if d.n >= 8
# ]
# apply_outer_speed_ratio(rotor, outer_elements, SPEED_RATIO)


# Campbell diagram (x axis: inner rotor speed)
# speed_range = Q_(np.linspace(0, 15000, 81), "RPM").to("rad/s").m
# campbell = rotor.run_campbell(speed_range, frequencies=10)
# fig1 = campbell.plot(
#     harmonics=[1, SPEED_RATIO], speed_units="RPM", frequency_units="Hz"
# )
# fig1.update_yaxes(range=(0, 400))
# fig1.show()


# def outer_unbalance_response(rotor, node, magnitude, phase, speed_range, ratio):
#     """Compute the response to an unbalance on the outer rotor.

#     The unbalance rotates with the outer rotor, so the excitation frequency is
#     ``ratio`` times the inner rotor speed.
#     """
#     frequency_range = ratio * speed_range
#     force = rotor._unbalance_force(node, magnitude, phase, frequency_range)

#     forced_resp = np.zeros((rotor.ndof, len(speed_range)), dtype=complex)
#     for i, (speed, frequency) in enumerate(zip(speed_range, frequency_range)):
#         forced_resp[:, i] = (
#             rotor.transfer_matrix(speed=speed, frequency=frequency) @ force[:, i]
#         )

#     return ForcedResponseResults(
#         rotor=rotor,
#         forced_resp=forced_resp,
#         velc_resp=1j * frequency_range * forced_resp,
#         accl_resp=-(frequency_range**2) * forced_resp,
#         speed_range=speed_range,
#         unbalance=np.array([[node, magnitude, phase]]),
#     )


# unbalance_speed = Q_(np.linspace(0, 15000, 2001), "RPM").to("rad/s").m
# probes = [rs.Probe(d.n, Q_(45, "deg"), tag=d.tag) for d in disks]

# response_d1 = rotor.run_unbalance_response(
#     node=1, unbalance_magnitude=1e-4, unbalance_phase=0, speed_range=unbalance_speed
# )
# fig2 = response_d1.plot_magnitude(probe=probes, frequency_units="RPM").update_layout(
#     title="Unbalance 0.0001 kg.m on d1 (inner rotor)", yaxis_type="log"
# )

# response_d2 = outer_unbalance_response(
#     rotor,
#     node=9,
#     magnitude=1e-4,
#     phase=0,
#     speed_range=unbalance_speed,
#     ratio=SPEED_RATIO,
# )
# response_d2 = rotor.run_unbalance_response(
#     node=9,
#     unbalance_magnitude=1e-4,
#     unbalance_phase=0,
#     speed_range=unbalance_speed,
# )
# fig3 = response_d2.plot_magnitude(probe=probes, frequency_units="RPM").update_layout(
#     title="Unbalance 0.0001 kg.m on d2 (outer rotor, 1.5x inner speed)",
#     yaxis_type="log",
# )

# for i, style in enumerate(["solid", "dashdot", "dot", "dash"]):
#     fig2.data[i].line.dash = style
#     fig3.data[i].line.dash = style

# for fig in (fig2, fig3):
#     fig.update_xaxes(tickvals=[0, 5000, 10000, 15000])
#     fig.update_yaxes(range=[-7, -1], tickvals=[1e-6, 1e-4, 1e-2], exponentformat="power")

# fig2.show()
# fig3.show()
