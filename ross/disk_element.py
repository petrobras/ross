import numpy as np
import matplotlib.patches as mpatches
from ross.element import Element


__all__ = [
    "DiskElement",
]


class DiskElement(Element):
    """A disk element.
     This class will create a disk element from input data of inertia and mass.
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
     >>> disk = DiskElement(0, 32.58972765, 0.17808928, 0.32956362)
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
        >>> disk = DiskElement(0, 32.58972765, 0.17808928, 0.32956362)
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

    def K(self):
        K = np.zeros((4, 4))

        return K

    def C(self):
        C = np.zeros((4, 4))

        return C

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
        >>> disk = DiskElement(0, 32.58972765, 0.17808928, 0.32956362)
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
    def from_geometry(cls, n, material, width, i_d, o_d):
        """A disk element.
        This class will create a disk element from input data of geometry.
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
        >>> disk = DiskElement.from_geometry(0, steel, 0.07, 0.05, 0.28)
        >>> disk.Ip
        0.32956362089137037
        """
        m = 0.25 * material.rho * np.pi * width * (o_d ** 2 - i_d ** 2)
        Id = (
            0.015625 * material.rho * np.pi * width * (o_d ** 4 - i_d ** 4)
            + m * (width ** 2) / 12
            )
        Ip = 0.03125 * material.rho * np.pi * width * (o_d ** 4 - i_d ** 4)

        return cls(n, m, Id, Ip)
