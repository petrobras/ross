import numpy as np
from ross import SealElement, HolePatternSeal, LabyrinthSeal
from ross.units import check_units, Q_


class HybridSeal(SealElement):
    """Hybrid seal - Compressible flow model with rotordynamic coefficients.

    This class provides a **comprehensive analytical model** for hybrid seals
    based on compressible gas flow through multiple throttling stages (teeth). The
    model calculates leakage rates and dynamic coefficients for rotordynamic analysis.

    **Theoretical Approach:**

    The model solves the **1D compressible flow problem** through a series of teeth using:
    """

    @check_units
    def __init__(
        self,
        n=None,
        # Parameters for LabyrinthSeal
        inlet_pressure=None,
        outlet_pressure=None,
        inlet_temperature=None,
        pre_swirl_ratio=None,
        frequency=None,
        n_teeth=None,
        shaft_radius=None,
        radial_clearance=None,
        pitch=None,
        tooth_height=None,
        tooth_width=None,
        seal_type=None,
        gas_composition=None,
        r=None,
        gamma=None,
        tz=None,
        muz=None,
        analz="FULL",
        nprt=1,
        iopt1=0,
        # Parameters for HolePatternSeal
        length=None,
        radius=None,
        clearance=None,
        roughness=None,
        cell_length=None,
        cell_width=None,
        cell_depth=None,
        b_suther=None,
        s_suther=None,
        molar=None,
        preswirl=None,
        entr_coef=None,
        exit_coef=None,
        nz=80,
        itrmx=180,
        stopcriterion=0.0001,
        toler=0.01,
        rlx=0.1,
        whirl_ratio=1.0,
        tolerance=1e-6,
        max_iterations=1e20,
        print_results=False,
        color="#787FF6",
        scale_factor=0.75,
        **kwargs,
    ):
        p_low = outlet_pressure
        p_high = inlet_pressure
        iteration = 0
        convergence_mdot = 1

        while convergence_mdot > tolerance and iteration < max_iterations:
            intermediate_pressure = (p_low + p_high) / 2

            laby = LabyrinthSeal(
                n=n,
                inlet_pressure=inlet_pressure,
                outlet_pressure=intermediate_pressure,
                inlet_temperature=inlet_temperature,
                pre_swirl_ratio=pre_swirl_ratio,
                frequency=frequency,
                n_teeth=n_teeth,
                shaft_radius=shaft_radius,
                radial_clearance=radial_clearance,
                pitch=pitch,
                tooth_height=tooth_height,
                tooth_width=tooth_width,
                seal_type=seal_type,
                gas_composition=gas_composition,
                r=r,
                gamma=gamma,
                tz=tz,
                muz=muz,
                analz=analz,
                nprt=nprt,
                iopt1=iopt1,
            )

            hole = HolePatternSeal(
                n=n,
                inlet_pressure=intermediate_pressure,
                outlet_pressure=outlet_pressure,
                inlet_temperature=inlet_temperature,
                frequency=frequency,
                length=length,
                radius=radius,
                clearance=clearance,
                roughness=roughness,
                cell_length=cell_length,
                cell_width=cell_width,
                cell_depth=cell_depth,
                gas_composition=gas_composition,
                b_suther=b_suther,
                s_suther=s_suther,
                molar=molar,
                gamma=gamma,
                preswirl=preswirl,
                entr_coef=entr_coef,
                exit_coef=exit_coef,
                nz=nz,
                itrmx=itrmx,
                stopcriterion=stopcriterion,
                toler=toler,
                rlx=rlx,
                whirl_ratio=whirl_ratio,
            )

            convergence_mdot = abs(hole.mdot - laby.mdot) / laby.mdot

            if laby.mdot > hole.mdot:
                p_low = intermediate_pressure
            else:
                p_high = intermediate_pressure

            iteration += 1

            print(
                f"{iteration:0.0f} | {convergence_mdot:.9e} | {intermediate_pressure:.9e} | {laby.mdot:.9e} | {hole.mdot:.9e}"
            )

        kxx = list(np.array(laby.kxx) + np.array(hole.kxx))
        kyy = list(np.array(laby.kyy) + np.array(hole.kyy))
        kxy = list(np.array(laby.kxy) + np.array(hole.kxy))
        kyx = list(np.array(laby.kyx) + np.array(hole.kyx))
        cxx = list(np.array(laby.cxx) + np.array(hole.cxx))
        cyy = list(np.array(laby.cyy) + np.array(hole.cyy))
        cxy = list(np.array(laby.cxy) + np.array(hole.cxy))
        cyx = list(np.array(laby.cyx) + np.array(hole.cyx))

        super().__init__(
            n,
            frequency=frequency,
            kxx=kxx,
            kyy=kyy,
            kxy=kxy,
            kyx=kyx,
            cxx=cxx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
            color=color,
            scale_factor=scale_factor,
            **kwargs,
        )


# hybrid_seal = HybridSeal(
#     n=0,

#     # Parâmetros compartilhados
#     inlet_pressure=500000.0,            # 5 bar entrada
#     outlet_pressure=100000.0,           # 1 bar saída
#     inlet_temperature=300.0,            # 300 K (27°C)
#     frequency=Q_([6000, 8000, 10000], "RPM"),
#     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},

#     # LabyrinthSeal - Primeira seção (throttling)
#     n_teeth=12,
#     shaft_radius=Q_(75, "mm"),
#     radial_clearance=Q_(0.25, "mm"),
#     pitch=Q_(3.0, "mm"),
#     tooth_height=Q_(3.0, "mm"),
#     tooth_width=Q_(0.15, "mm"),
#     seal_type="inter",                  # Interlocking
#     pre_swirl_ratio=0.95,

#     # HolePatternSeal - Segunda seção (dampening)
#     length=0.050,                       # 50 mm
#     radius=0.075,                       # 75 mm (mesmo que shaft_radius)
#     clearance=0.0003,                   # 0.3 mm
#     roughness=0.0001,
#     cell_length=0.003,
#     cell_width=0.003,
#     cell_depth=0.002,
#     preswirl=0.8,
#     entr_coef=0.5,
#     exit_coef=1.0,
#     nz=20,            # Para debug
# )


# import ross as rs
# from ross.seals.hybrid_seal import HybridSeal
# from ross.units import Q_
# import numpy as np

# # Criar eixo
# steel = rs.materials.steel
# shaft = [rs.ShaftElement(0.25, 0, 0.05, material=steel) for _ in range(6)]

# # Criar discos
# disk0 = rs.DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
# disk1 = rs.DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.28)

# # Criar mancais
# bearing0 = rs.BearingElement(0, kxx=1e6, cxx=1e3)
# bearing1 = rs.BearingElement(6, kxx=1e6, cxx=1e3)

# # Criar selo híbrido
# hybrid_seal = HybridSeal(
#     n=3,
#     inlet_pressure=500000,
#     outlet_pressure=100000,
#     inlet_temperature=300,
#     frequency=Q_([5000], "RPM"),
#     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
#     # LabyrinthSeal params
#     n_teeth=10,
#     shaft_radius=Q_(25, "mm"),
#     radial_clearance=Q_(0.25, "mm"),
#     pitch=Q_(3, "mm"),
#     tooth_height=Q_(3, "mm"),
#     tooth_width=Q_(0.15, "mm"),
#     seal_type="inter",
#     pre_swirl_ratio=0.9,
#     # HolePatternSeal params
#     length=0.04,
#     radius=0.025,
#     clearance=0.0003,
#     roughness=0.0001,
#     cell_length=0.003,
#     cell_width=0.003,
#     cell_depth=0.002,
#     preswirl=0.8,
#     entr_coef=0.5,
#     exit_coef=1.0,
# )

# # Montar rotor
# rotor = rs.Rotor(shaft, [disk0, disk1], [bearing0, bearing1, hybrid_seal])

# rotor.plot_rotor().show()
# print("dd")
