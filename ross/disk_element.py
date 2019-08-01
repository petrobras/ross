import bokeh.palettes as bp
from bokeh.models import ColumnDataSource, HoverTool
import matplotlib.patches as mpatches
import numpy as np
import toml
import pandas as pd
import sys
import warnings
import xlrd

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

    def patch(self, position, length, ax):
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

    def bokeh_patch(self, position, length, bk_ax):
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
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.
        """
        zpos, ypos = position
        le = length / 8

        # bokeh plot - coordinates to plot disks elements
        z_upper = [zpos, zpos + le, zpos - le]
        y_upper = [ypos, ypos * 4, ypos * 4]

        z_lower = [zpos, zpos + le, zpos - le]
        y_lower = [-ypos, -ypos * 4, -ypos * 4]

        source = ColumnDataSource(
            dict(z_l=z_lower, y_l=y_lower, z_u=z_upper, y_u=y_upper)
        )
        source_c = ColumnDataSource(
            dict(
                z_circle=[z_upper[0]],
                yu_circle=[y_upper[1]],
                yl_circle=[-y_upper[1]],
                radius=[le],
                elnum=[self.n],
                IP=[self.Ip],
                ID=[self.Id],
                mass=[self.m],
            )
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

        hover = HoverTool(names=["uc_disk", "lc_disk"])
        hover.tooltips = [
            ("Disk Node :", "@elnum"),
            ("Polar Moment of Inertia :", "@IP"),
            ("Diametral Moment of Inertia :", "@ID"),
            ("Disk mass :", "@mass"),
        ]
        hover.mode = "mouse"

        if len(bk_ax.hover) == 1:
            bk_ax.add_tools(hover)

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

    @classmethod
    def from_table(cls, file, sheet_name=0):
        """Instantiate one or more disks using inputs from a table, either excel or csv.
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
        is_csv = False
        try:
            df = pd.read_excel(file, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            sys.exit(file + " not found.")
        except xlrd.biffh.XLRDError:
            df = pd.read_csv(file)
            is_csv = True
        header_index = -1
        header_found = False
        for index, row in df.iterrows():
            for i in range(0, row.size):
                if isinstance(row[i], str):
                    if row[i].lower() == 'ip':
                        header_index = index
                        header_found = True
                        break
            if header_found:
                break
        if header_index < 0:
            sys.exit("Could not find the header. Make sure the sheet has a header "
                     "containing the names of the columns.")
        if not is_csv:
            df = pd.read_excel(file, header=header_index, sheet_name=sheet_name)
            df_unit = pd.read_excel(file, header=header_index, nrows=2, sheet_name=sheet_name)
        else:
            df = pd.read_csv(file, header=header_index)
            df_unit = pd.read_csv(file, header=header_index, nrows=2)
        convert_to_metric = True
        for index, row in df_unit.iterrows():
            for i in range(0, row.size):
                if isinstance(row[i], str):
                    if 'kg' in row[i].lower():
                        convert_to_metric = False
                        break
            if not convert_to_metric:
                break
        first_data_row = -1
        for index, row in df.iterrows():
            if isinstance(row[0], int) or isinstance(row[0], float):
                first_data_row = index
                break
        if first_data_row < 0:
            sys.exit("Could not find the data. Make sure you have at least one row containing "
                     "data below the header.")
        for i in range(0, first_data_row):
            df = df.drop(i)
        nan_found = False
        for index, row in df.iterrows():
            for i in range(first_data_row, row.size):
                if pd.isna(row[i]):
                    nan_found = True
                    row[i] = 0
        if nan_found:
            warnings.warn("One or more NaN found. They were replaced with zeros.")
        parameters = []
        possible_names = [["Unnamed: 0", "n", "N"], ["m", "M", "mass", "Mass", "MASS"],
                          ["ip", "Ip", "IP"],
                          ["it", "It", "IT", "id", "Id", "ID"]]
        for name_group in possible_names:
            for name in name_group:
                try:
                    parameter = df[name].tolist()
                    parameters.append(parameter)
                    break
                except KeyError:
                    continue
        disk_list = []
        convert_factor = [1, 1]
        if convert_to_metric:
            convert_factor[0] = 0.45359237
            convert_factor[1] = 0.0002926397
        for i in range(0, len(parameters[0])):
            disk_list.append(cls(n=parameters[0][i],
                                 m=parameters[1][i]*convert_factor[0],
                                 Ip=parameters[2][i]*convert_factor[1],
                                 Id=parameters[3][i]*convert_factor[1]))
        return disk_list
