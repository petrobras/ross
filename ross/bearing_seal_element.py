import warnings
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ross.element import Element

__all__ = [
    "BearingElement",
    "SealElement",
]
   

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

    def M(self):
        M = np.zeros((4, 4))

        return M

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
