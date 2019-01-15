import warnings
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abc import ABC

__all__ = [
    "BearingElement",
    "SealElement",
    "IsotSealElement",
]

class Element(ABC):
    """Element class."""

    def __init__(self):
        pass

    def summary(self):
        """A summary for the element.
        A pandas series with the element properties as variables.
        """
        attributes = self.__dict__
        attributes["type"] = self.__class__.__name__
        return pd.Series(attributes)

    # These are the abstract classes for mass, damping and stiffness matrices
    def M(self):
        pass
    
    def C(self):
        pass
    
    def K(self):
        pass

class _Coefficient:
    def __init__(self, coefficient, w=None, interpolated=None):
        if isinstance(coefficient, (int, float)):
            if w is not None:
                coefficient = [coefficient for _ in range(len(w))]
            else:
                coefficient = [coefficient]

        self.coefficient = coefficient
        self.w = w

        if len(self.coefficient) > 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.interpolated = interpolate.UnivariateSpline(
                        self.w, self.coefficient
                    )
            #  dfitpack.error is not exposed by scipy
            #  so a bare except is used
            except:
                raise ValueError(
                    "Arguments (coefficients and w)" " must have the same dimension"
                )
        else:
            self.interpolated = lambda x: np.array(self.coefficient[0])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        w_range = np.linspace(min(self.w), max(self.w), 30)

        ax.plot(w_range, self.interpolated(w_range), **kwargs)
        ax.set_xlabel("Speed (rad/s)")
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

        return ax



class _Stiffness_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel("Stiffness ($N/m$)")

        return ax



class _Damping_Coefficient(_Coefficient):
    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_ylabel("Damping ($Ns/m$)")

        return ax



class BearingElement(Element):

    """A bearing element.
    This class will create a bearing element.
    Parameters can be a constant value or speed dependent.
    For speed dependent parameters, each argument should be passed
    as an array and the correspondent speed values should also be
    passed as an array.
    Values for each parameter will be interpolated for the speed.
    Parameters
    ----------
    kxx: float, array
        Direct stiffness in the x direction.
    cxx: float, array
        Direct damping in the x direction.
    kyy: float, array, optional
        Direct stiffness in the y direction.
        (defaults to kxx)
    cyy: float, array, optional
        Direct damping in the y direction.
        (defaults to cxx)
    kxy: float, array, optional
        Cross coupled stiffness in the x direction.
        (defaults to 0)
    cxy: float, array, optional
        Cross coupled damping in the x direction.
        (defaults to 0)
    kyx: float, array, optional
        Cross coupled stiffness in the y direction.
        (defaults to 0)
    cyx: float, array, optional
        Cross coupled damping in the y direction.
        (defaults to 0)
    w: array, optional
        Array with the speeds (rad/s).
    Examples
    --------
    """


    def __init__(
        self, n, kxx, cxx, kyy=None, kxy=0, kyx=0, cyy=None, cxy=0, cyx=0, w=None
    ):

        args = ["kxx", "kyy", "kxy", "kyx", "cxx", "cyy", "cxy", "cyx"]

        # all args to coefficients
        args_dict = locals()
        coefficients = {}

        if kyy is None:
            args_dict["kyy"] = kxx
        if cyy is None:
            args_dict["cyy"] = cxx

        for arg in args:
            if arg[0] == "k":
                coefficients[arg] = _Stiffness_Coefficient(
                    args_dict[arg], args_dict["w"]
                )
            else:
                coefficients[arg] = _Damping_Coefficient(args_dict[arg], args_dict["w"])

        coefficients_len = [len(v.coefficient) for v in coefficients.values()]

        if w is not None:
            coefficients_len.append(len(args_dict["w"]))
            if len(set(coefficients_len)) > 1:
                raise ValueError(
                    "Arguments (coefficients and w)" " must have the same dimension"
                )
        else:
            for c in coefficients_len:
                if c != 1:
                    raise ValueError(
                        "Arguments (coefficients and w)" " must have the same dimension"
                    )

        for k, v in coefficients.items():
            setattr(self, k, v)

        self.n = n
        self.n_l = n
        self.n_r = n

        self.w = np.array(w, dtype=np.float64)
        self.color = "#355d7a"

    def __repr__(self):
        return "%s" % self.__class__.__name__

    def K(self, w):
        kxx = self.kxx.interpolated(w)
        kyy = self.kyy.interpolated(w)
        kxy = self.kxy.interpolated(w)
        kyx = self.kyx.interpolated(w)

        K = np.array([[kxx, kxy], [kyx, kyy]])

        return K

    def C(self, w):
        cxx = self.cxx.interpolated(w)
        cyy = self.cyy.interpolated(w)
        cxy = self.cxy.interpolated(w)
        cyx = self.cyx.interpolated(w)

        C = np.array([[cxx, cxy], [cyx, cyy]])

        return C

    def patch(self, ax, position):
        """Bearing element patch.
        Patch that will be used to draw the bearing element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position (z, y) in which the patch will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        h = -0.75 * ypos  # height

        #  node (x pos), outer diam. (y pos)
        bearing_points = [
            [zpos, ypos],  # upper
            [zpos + h / 2, ypos - h],
            [zpos - h / 2, ypos - h],
            [zpos, ypos],
        ]
        ax.add_patch(mpatches.Polygon(bearing_points, color=self.color, picker=True))

class SealElement(BearingElement):
    def __init__(
        self,
        n,
        kxx,
        cxx,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        w=None,
        seal_leakage=None,
    ):
        super().__init__(
            n=n,
            w=w,
            kxx=kxx,
            kxy=kxy,
            kyx=kyx,
            kyy=kyy,
            cxx=cxx,
            cxy=cxy,
            cyx=cyx,
            cyy=cyy,
        )

        self.seal_leakage = seal_leakage
        self.color = "#77ACA2"

    def patch(self, ax, position):
        """Seal element patch.
        Patch that will be used to draw the seal element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : tuple
            Position in which the patch will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        hw = 0.05

        #  node (x pos), outer diam. (y pos)
        seal_points_u = [
            [zpos, ypos * 1.1],  # upper
            [zpos + hw, ypos * 1.1],
            [zpos + hw, ypos * 1.3],
            [zpos - hw, ypos * 1.3],
            [zpos - hw, ypos * 1.1],
            [zpos, ypos * 1.1],
        ]
        seal_points_l = [
            [zpos, -ypos * 1.1],  # lower
            [zpos + hw, -(ypos * 1.1)],
            [zpos + hw, -(ypos * 1.3)],
            [zpos - hw, -(ypos * 1.3)],
            [zpos - hw, -(ypos * 1.1)],
            [zpos, -ypos * 1.1],
        ]
        ax.add_patch(mpatches.Polygon(seal_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(seal_points_l, facecolor=self.color))


class IsotSealElement(SealElement):
    def __init__(
        self,
        n,
        kxx,
        cxx,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        w=None,
        exit_axial_mach_number=None,
        exit_axial_reynolds_number=None,
        exit_circumferential_reynolds_number=None,
        kxx_fd=None,
        cxx_fd=None,
        kyy_fd=None,
        kxy_fd=None,
        kyx_fd=None,
        cyy_fd=None,
        cxy_fd=None,
        cyx_fd=None,
        w_fd=None,
        wfr=None,
        seal_leakage=None,
        absolute_viscosity=None,
        cell_vol_to_area_ratio=None,
        compressibility_factor=None,
        entrance_loss_coefficient=None,
        exit_clearance=None,
        exit_recovery_factor=None,
        inlet_clearance=None,
        inlet_preswirl_ratio=None,
        molecular_weight=None,
        number_integr_steps=None,
        p_exit=None,
        p_supply=None,
        reservoir_temperature=None,
        seal_diameter=None,
        seal_length=None,
        specific_heat_ratio=None,
        speed=None,
        tolerance_percentage=None,
        turbulence_coef_mr=None,
        turbulence_coef_ms=None,
        turbulence_coef_nr=None,
        turbulence_coef_ns=None,
    ):
        super().__init__(
            n=n,
            w=w,
            kxx=kxx,
            kxy=kxy,
            kyx=kyx,
            kyy=kyy,
            cxx=cxx,
            cxy=cxy,
            cyx=cyx,
            cyy=cyy,
            seal_leakage=seal_leakage,
        )

        self.absolute_viscosity = absolute_viscosity
        self.cell_vol_to_area_ratio = cell_vol_to_area_ratio
        self.compressibility_factor = compressibility_factor
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_clearance = exit_clearance
        self.exit_recovery_factor = exit_recovery_factor
        self.inlet_clearance = inlet_clearance
        self.inlet_preswirl_ratio = inlet_preswirl_ratio
        self.molecular_weight = molecular_weight
        self.number_integr_steps = number_integr_steps
        self.p_exit = p_exit
        self.p_supply = p_supply
        self.reservoir_temperature = reservoir_temperature
        self.seal_diameter = seal_diameter
        self.seal_length = seal_length
        self.specific_heat_ratio = specific_heat_ratio
        self.speed = speed
        self.tolerance_percentage = tolerance_percentage
        self.turbulence_coef_mr = turbulence_coef_mr
        self.turbulence_coef_ms = turbulence_coef_ms
        self.turbulence_coef_nr = turbulence_coef_nr
        self.turbulence_coef_ns = turbulence_coef_ns
        self.exit_axial_mach_number = exit_axial_mach_number,
        self.exit_axial_reynolds_number = exit_axial_reynolds_number,
        self.exit_circumferential_reynolds_number = exit_circumferential_reynolds_number
        self.kxx_fd = kxx_fd
        self.cxx_fd = cxx_fd
        self.kyy_fd = kyy_fd
        self.kxy_fd = kxy_fd
        self.kyx_fd = kyx_fd
        self.cyy_fd = cyy_fd
        self.cxy_fd = cxy_fd
        self.cyx_fd = cyx_fd
        self.w_fd = w_fd
        self.wfr = wfr

    def kxx_eff(self):
        return (self.kxx.coefficient + self.cxy.coefficient
                * self.kxx.w)

    def cxx_eff(self):
        return (self.cxx.coefficient - self.kxy.coefficient
                / self.cxx.w)

    def kxx_fd_eff(self):
        return (self.kxx_fd + self.cxy_fd
                * self.w_fd)

    def cxx_fd_eff(self):
        return (self.cxx_fd - self.kxy_fd
                / self.w_fd)

    def effective_acoustic_velocity(self, gamma=0.69):
        zc = self.compressibility_factor  # dimensionless
        Rg = 8.3144598  # joule / (kelvin mole)
        T = self.reservoir_temperature + 273.15  # kelvin
        mw = self.molecular_weight / 1000  # kilogram / mole
        H = self.inlet_clearance / 1000  # meter
        Hd = self.cell_vol_to_area_ratio / 1000 / gamma  # meter
        c0 = ((zc * T * Rg/mw) / (1 + (Hd / H))) ** 0.5
        return c0
