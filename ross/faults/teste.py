from pathlib import Path
import sys
# sys.path.append("C:\\Users\\LMEst Emanuela\\OneDrive - Universidade Federal de Uberlândia\\lmest\\ambiente_virtual_2\\.venv\\ross-main\\ross-main\\ross") 
# sys.path.append("C:\\Users\\LMEst Emanuela\\OneDrive - Universidade Federal de Uberlândia\\lmest\\ambiente_virtual_2\\.venv\\ross\\ross") 
import ross as rs
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from ross.units import Q_

pio.renderers.default = "browser"


steel = rs.Material(name="Steel", rho = 7850, Poisson = 0.29, E = 2.05e11)

i_d = 0
o_d = 0.0170

L = np.array(
        [0, 25, 49, 65, 80, 110, 140, 161, 190, 220, 250, 285, 295, 305, 330, 360, 390, 
            420, 450, 480, 510, 523, 533, 543, 570, 594, 630, 664, 700, 730, 760, 790, 830, 862]
         )/ 1000

L = [L[i] - L[i - 1] for i in range(1, len(L))]

n = len(L) - 1

shaft_elem = [
    rs.ShaftElement(
        material=steel,
        L=l,
        idl=i_d,
        odl=o_d,
        idr=i_d,
        odr=o_d,
        alpha=2.730,
        beta=4.85e-4,
        rotary_inertia=True,
        shear_effects=True,
    )
    for l in L
]

n_list = [12, 22]
m_list = [2.637, 2.649]
Id_list = [0.003844540885417, 0.003844540885417]
Ip_list = [0.007513248437500, 0.007547431937500]
disk_elements = [
    rs.DiskElement(
        n=n,
        m=m,
        Id=Id,
        Ip=Ip,
    )
    for n, m, Id, Ip in zip(n_list, m_list, Id_list, Ip_list)
]

kxx1 = 8.551e5
kyy1 = 1.198e6
kzz = 1e15
cxx1 = 7.452
cyy1 = 33.679
czz1 = 300

kxx2 = 5.202e7
kyy2 = 7.023e8
kzz2 = 1e15
cxx2 = 25.587
cyy2 = 91.033
czz2 = 300

bearing0 = rs.BearingElement(
    n=3, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz1
)
bearing1 = rs.BearingElement(
    n=30, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz2
)

rotor = rs.Rotor(shaft_elem, disk_elements, [bearing0, bearing1])
fig = rotor.plot_rotor()
fig.show()


fault = rs.Crack(rotor, n=18, depth_ratio=0.5, crack_model="Flex Breathing", cross_discret=10)
probe3 = (27,0)

results = fault.run(
        node=[12, 22],
        unb_magnitude=[5e-4, 0],
        unb_phase=[-np.pi / 2, 0],
        speed=125.66,   # rad/s
        t=np.arange(0, 10, 0.00001),
    )

results.plot_1d([probe3]).show()

results.plot_dfft([probe3], frequency_range=(0,250*2*np.pi) , yaxis_type="log").show()