import bokeh.palettes as bp
from bokeh.models import ColumnDataSource, HoverTool
import matplotlib.patches as mpatches
import numpy as np
import toml
from ross.utils import read_table_file

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
     tag : str, optional
         A tag to name the element
         Default is None
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

    def __init__(self, n, m, Id, Ip, tag=None):
        self.n = int(n)
        self.n_l = n
        self.n_r = n

        self.m = m
        self.Id = Id
        self.Ip = Ip
        self.tag = tag
        self.color = "#bc625b"

    def __eq__(self, other):
        false_number = 0
        for i in self.__dict__:
            try:
                if np.allclose(self.__dict__[i], other.__dict__[i]):
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
            f"n={self.n}, tag={self.tag!r})"
        )

    def save(self, file_name):
        data = self.load_data(file_name)
        data["DiskElement"][str(self.n)] = {
            "n": self.n,
            "m": self.m,
            "Id": self.Id,
            "Ip": self.Ip,
            "tag": self.tag,
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

    def dof_mapping(self):
        return dict(x0=0, y0=1, alpha0=2, beta0=3)

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

    def patch(self, position, ax):
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
        step = ypos / 5

        #  matplotlib node (x pos), outer diam. (y pos)
        disk_points_u = [
            [zpos, ypos],  # upper
            [zpos + step, ypos * 4],
            [zpos - step, ypos * 4],
            [zpos, ypos],
        ]
        disk_points_l = [
            [zpos, -ypos],  # lower
            [zpos + step, -ypos * 4],
            [zpos - step, -ypos * 4],
            [zpos, -ypos],
        ]

        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))

        ax.add_patch(
            mpatches.Circle(xy=(zpos, ypos*4), radius=step, color=self.color)
        )
        ax.add_patch(
            mpatches.Circle(xy=(zpos, -ypos*4), radius=step, color=self.color)
        )

    def bokeh_patch(self, position, bk_ax):
        """Disk element patch.
        Patch that will be used to draw the disk element.
        Parameters
        ----------
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        position : float
            Position in which the patch will be drawn.
        Returns
        -------
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        step = ypos / 5

        # bokeh plot - coordinates to plot disks elements
        z_upper = [zpos, zpos + step, zpos - step]
        y_upper = [ypos, ypos * 4, ypos * 4]

        z_lower = [zpos, zpos + step, zpos - step]
        y_lower = [-ypos, -ypos * 4, -ypos * 4]

        source = ColumnDataSource(
            dict(
                z_l=[z_lower],
                y_l=[y_lower],
                z_u=[z_upper],
                y_u=[y_upper],
                elnum=[self.n],
                IP=[self.Ip],
                ID=[self.Id],
                mass=[self.m],
                tag=[self.tag],
            )
        )
        source_c = ColumnDataSource(
            dict(
                z_circle=[z_upper[0]],
                yu_circle=[y_upper[1]],
                yl_circle=[-y_upper[1]],
                radius=[step],
                elnum=[self.n],
                IP=[self.Ip],
                ID=[self.Id],
                mass=[self.m],
                tag=[self.tag],
            )
        )

        bk_ax.patches(
            xs="z_u",
            ys="y_u",
            source=source,
            alpha=1,
            line_width=2,
            color=bokeh_colors[9],
            legend="Disk",
            name="ub_disk",
        )
        bk_ax.patches(
            xs="z_l",
            ys="y_l",
            source=source,
            alpha=1,
            line_width=2,
            color=bokeh_colors[9],
            name="ub_disk",
        )
        bk_ax.circle(
            x="z_circle",
            y="yu_circle",
            radius="radius",
            source=source_c,
            fill_alpha=1,
            color=bokeh_colors[9],
            name="uc_disk",
        )
        bk_ax.circle(
            x="z_circle",
            y="yl_circle",
            radius="radius",
            source=source_c,
            fill_alpha=1,
            color=bokeh_colors[9],
            name="lc_disk",
        )

        hover = HoverTool(names=["uc_disk", "lc_disk", "ub_disk", "lb_disk"])
        hover.tooltips = [
            ("Disk Node :", "@elnum"),
            ("Polar Moment of Inertia :", "@IP"),
            ("Diametral Moment of Inertia :", "@ID"),
            ("Disk mass :", "@mass"),
            ("Tag :", "@tag")
        ]
        hover.mode = "mouse"

        return hover


    @classmethod
    def from_geometry(cls, n, material, width, i_d, o_d, tag=None):
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
        tag : str, optional
            A tag to name the element
            Default is None
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

        tag = tag

        return cls(n, m, Id, Ip, tag)

    @classmethod
    def from_table(cls, file, sheet_name=0):
        """Instantiate one or more disks using inputs from an Excel table.
        A header with the names of the columns is required. These names should match the names expected by the routine
        (usually the names of the parameters, but also similar ones). The program will read every row bellow the header
        until they end or it reaches a NaN.
        Parameters
        ----------
        file: str
            Path to the file containing the disk parameters.
        sheet_name: int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is passed, it is
            assumed to be the first sheet in the file.
        Returns
        -------
        disk : list
            A list of disk objects.
        """
        parameters = read_table_file(file, 'disk', sheet_name=sheet_name)
        list_of_disks = []
        for i in range(0, len(parameters['n'])):
            list_of_disks.append(cls(n=parameters['n'][i], m=parameters['m'][i],
                                     Id=parameters['Id'][i], Ip=parameters['Ip'][i]))
        return list_of_disks
