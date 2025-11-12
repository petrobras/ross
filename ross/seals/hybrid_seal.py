from ross import SealElement, HolePatternSeal, LabyrinthSeal
from ross.units import check_units


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
        print_results=False,
        **kwargs,
    ):
        laby = LabyrinthSeal(
            n=n,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure,
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

        holepattern = HolePatternSeal(
            n=n,
            frequency=frequency,
            length=length,
            radius=radius,
            clearance=clearance,
            roughness=roughness,
            cell_length=cell_length,
            cell_width=cell_width,
            cell_depth=cell_depth,
            inlet_pressure=laby.outlet_pressure,
            outlet_pressure=outlet_pressure,
            inlet_temperature=inlet_temperature,
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
