import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.patches as mpatches

__all__ = [
    "LumpedDiskElement",
    "DiskElement",
]

class Element:
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

class LumpedDiskElement(Element):
    """A lumped disk element.
     This class will create a lumped disk element.
     Parameters
     ----------
     n: int
         Node in which the disk will be inserted.
     m : float
         Mass of the disk element.
     Id : float
         Diametral moment of inertia.
     Ip : float
         Polar moment of inertia
     References
     ----------
     .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
        & AW Lees, published by Cambridge University Press, 2010 pp. 156-157.
     Examples
     --------
     >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
     >>> disk.Ip
     0.32956362
     """

    def __init__(self, n, m, Id, Ip):
        self.n = n
        self.n_l = n
        self.n_r = n

        self.m = m
        self.Id = Id
        self.Ip = Ip
        self.color = "#bc625b"

    def M(self):
        """
        This method will return the mass matrix for an instance of a disk
        element.
        Parameters
        ----------
        self
        Returns
        -------
        Mass matrix for the disk element.
        Examples
        --------
        >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.M()
        array([[ 32.58972765,   0.        ,   0.        ,   0.        ],
               [  0.        ,  32.58972765,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.17808928,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.17808928]])
        """
        m = self.m
        Id = self.Id
        # fmt: off
        M = np.array([[m, 0,  0,  0],
                       [0, m,  0,  0],
                       [0, 0, Id,  0],
                       [0, 0,  0, Id]])
        # fmt: on
        return M

    def G(self):
        """
        This method will return the gyroscopic matrix for an instance of a disk
        element.
        Parameters
        ----------
        self
        Returns
        -------
        Gyroscopic matrix for the disk element.
        Examples
        --------
        >>> disk = LumpedDiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.G()
        array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.32956362],
               [ 0.        ,  0.        , -0.32956362,  0.        ]])
        """

        Ip = self.Ip
        # fmt: off
        G = np.array([[0, 0,   0,  0],
                      [0, 0,   0,  0],
                      [0, 0,   0, Ip],
                      [0, 0, -Ip,  0]])
        # fmt: on
        return G

    def patch(self, ax, position):
        """Lumped Disk element patch.
        Patch that will be used to draw the disk element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        D = ypos * 1.5
        hw = 0.005

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [
            [zpos, ypos],  # upper
            [zpos + hw, ypos + D],
            [zpos - hw, ypos + D],
            [zpos, ypos],
        ]
        disk_points_l = [
            [zpos, -ypos],  # lower
            [zpos + hw, -(ypos + D)],
            [zpos - hw, -(ypos + D)],
            [zpos, -ypos],
        ]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))

        ax.add_patch(
            mpatches.Circle(xy=(zpos, ypos + D), radius=0.01, color=self.color)
        )
        ax.add_patch(
            mpatches.Circle(xy=(zpos, -(ypos + D)), radius=0.01, color=self.color)
        )

    @classmethod
    def load_from_xltrc(cls, file, sheet_name="More"):
        df = load_disks_from_xltrc(file, sheet_name)
        disks = [cls(d.n - 1, d.Mass, d.It, d.Ip) for _, d in df.iterrows()]

        return disks


class DiskElement(LumpedDiskElement):
    #  TODO detail this class attributes inside the docstring
    """A disk element.
    This class will create a disk element.
    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    material : lavirot.Material
         Shaft material.
    width: float
        The disk width.
    i_d: float
        Inner diameter.
    o_d: float
        Outer diameter.
    Attributes
    ----------
    m : float
        Mass of the disk element.
    Id : float
        Diametral moment of inertia.
    Ip : float
        Polar moment of inertia
    References
    ----------
    .. [1] 'Dynamics of Rotating Machinery' by MI Friswell, JET Penny, SD Garvey
       & AW Lees, published by Cambridge University Press, 2010 pp. 156-157.
    Examples
    --------
    >>> from ross.materials import steel
    >>> disk = DiskElement(0, steel, 0.07, 0.05, 0.28)
    >>> disk.Ip
    0.32956362089137037
    """

    #  TODO add __repr__ to the class
    def __init__(self, n, material, width, i_d, o_d):
        if not isinstance(n, int):
            raise TypeError(f"n should be int, not {n.__class__.__name__}")
        self.n = n
        self.n_l = n
        self.n_r = n

        self.material = material
        self.rho = material.rho
        self.width = width

        # diameters
        self.i_d = i_d
        self.o_d = o_d
        self.i_d_l = i_d
        self.o_d_l = o_d
        self.i_d_r = i_d
        self.o_d_r = o_d

        self.m = 0.25 * self.rho * np.pi * width * (o_d ** 2 - i_d ** 2)
        self.Id = (
            0.015625 * self.rho * np.pi * width * (o_d ** 4 - i_d ** 4)
            + self.m * (width ** 2) / 12
        )
        self.Ip = 0.03125 * self.rho * np.pi * width * (o_d ** 4 - i_d ** 4)
        self.color = "#bc625b"

        super().__init__(self.n, self.m, self.Id, self.Ip)

    def patch(self, ax, position):
        """Disk element patch.
        Patch that will be used to draw the disk element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        if isinstance(position, tuple):
            position = position[0]
        zpos = position
        ypos = self.i_d
        D = self.o_d
        hw = self.width / 2  # half width

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [
            [zpos, ypos],  # upper
            [zpos + hw, ypos + 0.1 * D],
            [zpos + hw, ypos + 0.9 * D],
            [zpos - hw, ypos + 0.9 * D],
            [zpos - hw, ypos + 0.1 * D],
            [zpos, ypos],
        ]
        disk_points_l = [
            [zpos, -ypos],  # lower
            [zpos + hw, -(ypos + 0.1 * D)],
            [zpos + hw, -(ypos + 0.9 * D)],
            [zpos - hw, -(ypos + 0.9 * D)],
            [zpos - hw, -(ypos + 0.1 * D)],
            [zpos, -ypos],
        ]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))


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

