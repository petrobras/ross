import bokeh.palettes as bp
import matplotlib.patches as mpatches
import numpy as np
import pytest
import toml
from bokeh.models import ColumnDataSource

from ross.element import Element

__all__ = ["DiskElement"]
bokeh_colors = bp.RdGy[11]


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
        self.n = int(n)
        self.n_l = n
        self.n_r = n

        self.m = m
        self.Id = Id
        self.Ip = Ip
        self.color = "#bc625b"

    def __eq__(self, other):
        false_number = 0
        for i in self.__dict__:
            try:
                if pytest.approx(self.__dict__[i]) == other.__dict__[i]:
                    pass
                else:
                    false_number += 1

            except TypeError:
                if self.__dict__[i] == other.__dict__[i]:
                    pass
                else:
                    false_number += 1

        if false_number == 0:
            return True
        else:
            return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(Id={self.Id:{0}.{5}}, Ip={self.Ip:{0}.{5}}, "
            f"m={self.m:{0}.{5}}, color={self.color!r}, "
            f"n={self.n})"
        )

    def save(self, file_name):
        data = self.load_data(file_name)
        data["DiskElement"][str(self.n)] = {
            "n": self.n,
            "m": self.m,
            "Id": self.Id,
            "Ip": self.Ip,
        }
        self.dump_data(data, file_name)

    @staticmethod
    def load(file_name="DiskElement"):
        disk_elements = []
        with open("DiskElement.toml", "r") as f:
            disk_elements_dict = toml.load(f)
            for element in disk_elements_dict["DiskElement"]:
                disk_elements.append(
                    DiskElement(**disk_elements_dict["DiskElement"][element])
                )
        return disk_elements

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
        array([[32.58972765,  0.        ,  0.        ,  0.        ],
               [ 0.        , 32.58972765,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.17808928,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.17808928]])
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

    def patch(self, position, length, ax, bk_ax):
        """Disk element patch.
        Patch that will be used to draw the disk element.
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        length : float
            minimum length of shaft elements
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        le = length / 8
        D = ypos * 2
        hw = 0.02

        #  matplotlib node (x pos), outer diam. (y pos)
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

        ax.add_patch(mpatches.Circle(xy=(zpos, ypos + D), radius=hw, color=self.color))
        ax.add_patch(
            mpatches.Circle(xy=(zpos, -(ypos + D)), radius=hw, color=self.color)
        )

        # bokeh plot - coordinates to plot disks elements
        z_upper = [zpos, zpos + le, zpos - le]
        y_upper = [ypos, ypos * 4, ypos * 4]

        z_lower = [zpos, zpos + le, zpos - le]
        y_lower = [-ypos, -ypos * 4, -ypos * 4]

        z_circle = z_upper[0]
        y_circle = y_upper[1]

        source = ColumnDataSource(
            dict(z_l=z_lower, y_l=y_lower, z_u=z_upper, y_u=y_upper)
        )

        bk_ax.patch(
            x="z_u",
            y="y_u",
            source=source,
            alpha=1,
            line_width=2,
            color=bokeh_colors[9],
            legend="Disk",
        )
        bk_ax.patch(
            x="z_l",
            y="y_l",
            source=source,
            alpha=1,
            line_width=2,
            color=bokeh_colors[9],
        )
        bk_ax.circle(
            x=z_circle, y=y_circle, radius=le, fill_alpha=1, color=bokeh_colors[9]
        )
        bk_ax.circle(
            x=z_circle, y=-y_circle, radius=le, fill_alpha=1, color=bokeh_colors[9]
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
        # fmt: off
        Id = (
            0.015625 * material.rho * np.pi * width * (o_d ** 4 - i_d ** 4)
            + m * (width ** 2) / 12
        )
        # fmt: on
        Ip = 0.03125 * material.rho * np.pi * width * (o_d ** 4 - i_d ** 4)

        return cls(n, m, Id, Ip)
