"""Disk Element module.

This module defines the DiskElement classes which will be used to represent equipments
attached to the rotor shaft, which add mainly mass and inertia to the system.
There're 2 options, an element with 4 or 6 degrees of freedom.
"""
import os
from pathlib import Path

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import numpy as np
import toml
from bokeh.models import ColumnDataSource, HoverTool

from ross.element import Element
from ross.units import check_units
from ross.utils import read_table_file

__all__ = ["DiskElement", "DiskElement6DoF"]
bokeh_colors = bp.RdGy[11]


class DiskElement(Element):
    """A disk element.

    This class creates a disk element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, pint.Quantity
        Mass of the disk element.
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float, optional
        The scale factor is used to scale the disk drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#b2182b' (Cardinal).

    Examples
    --------
    >>> disk = DiskElement(n=0, m=32, Id=0.2, Ip=0.3)
    >>> disk.Ip
    0.3
    """

    @check_units
    def __init__(self, n, m, Id, Ip, tag=None, scale_factor=1.0, color=bokeh_colors[9]):
        self.n = int(n)
        self.n_l = n
        self.n_r = n

        self.m = m
        self.Id = Id
        self.Ip = Ip
        self.tag = tag
        self.color = color
        self.scale_factor = scale_factor
        self.dof_global_index = None

    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other: object
            The second object to be compared with.

        Returns
        -------
        bool
            True if the comparison is true; False otherwise.

        Examples
        --------
        >>> disk1 = disk_example()
        >>> disk2 = disk_example()
        >>> disk1 == disk2
        True
        """
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
        """Return a string representation of a disk element.

        Returns
        -------
        A string representation of a disk element object.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk # doctest: +ELLIPSIS
        DiskElement(Id=0.17809, Ip=0.32956...
        """
        return (
            f"{self.__class__.__name__}"
            f"(Id={self.Id:{0}.{5}}, Ip={self.Ip:{0}.{5}}, "
            f"m={self.m:{0}.{5}}, color={self.color!r}, "
            f"n={self.n}, tag={self.tag!r})"
        )

    def __hash__(self):
        return hash(self.tag)

    def save(self, file_name=os.getcwd()):
        """Save a disk element in a toml format.

        It works as an auxiliary function of the save function in the Rotor class.

        Parameters
        ----------
        file_name : string
            The name of the file the disk element will be saved in.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.save()
        """
        data = self.get_data(Path(file_name) / "DiskElement.toml")
        data["DiskElement"][str(self.n)] = {
            "n": self.n,
            "m": self.m,
            "Id": self.Id,
            "Ip": self.Ip,
            "tag": self.tag,
        }
        self.dump_data(data, Path(file_name) / "DiskElement.toml")

    @staticmethod
    def load(file_name=os.getcwd()):
        """Load a list of disk elements saved in a toml format.

        It works as an auxiliary function of the load function in the Rotor class.

        Parameters
        ----------
        file_name: str
            The name of the file of the disk element to be loaded.

        Returns
        -------
        disk_elements: list
            A list of disk elements.

        Examples
        --------
        >>> disk1 = disk_example()
        >>> disk1.save(os.getcwd())
        >>> list_of_disks = DiskElement.load(os.getcwd())
        >>> disk1 == list_of_disks[0]
        True
        """
        disk_elements = []
        with open("DiskElement.toml", "r") as f:
            disk_elements_dict = toml.load(f)
            for element in disk_elements_dict["DiskElement"]:
                disk_elements.append(
                    DiskElement(**disk_elements_dict["DiskElement"][element])
                )
        return disk_elements

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        The numbering of the degrees of freedom for each node.

        Being the following their ordering for a node:

        x_0 - horizontal translation
        y_0 - vertical translation
        alpha_0 - rotation around horizontal
        beta_0  - rotation around vertical

        >>> disk = disk_example()
        >>> disk.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'alpha_0': 2, 'beta_0': 3}
        """
        return dict(x_0=0, y_0=1, alpha_0=2, beta_0=3)

    def M(self):
        """Mass matrix for an instance of a disk element.

        This method will return the mass matrix for an instance of a disk element.

        Returns
        -------
        M : np.ndarray
            A matrix of floats containing the values of the mass matrix.

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
        """Stiffness matrix for an instance of a disk element.

        This method will return the stiffness matrix for an instance of a disk
        element.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.K()
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        K = np.zeros((4, 4))

        return K

    def C(self):
        """Damping matrix for an instance of a disk element.

        This method will return the damping matrix for an instance of a disk
        element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.C()
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C = np.zeros((4, 4))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a disk element.

        This method will return the gyroscopic matrix for an instance of a disk
        element.

        Returns
        -------
        G: np.ndarray
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

        Patch that will be used to draw the disk element using Matplotlib library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        """
        zpos, ypos, step = position
        radius = step / 6

        #  matplotlib node (x pos), outer diam. (y pos)
        disk_points_u = [
            [zpos, ypos],  # upper
            [zpos + step / 6, ypos + 2 * step],
            [zpos - step / 6, ypos + 2 * step],
            [zpos, ypos],
        ]
        disk_points_l = [
            [zpos, -ypos],  # lower
            [zpos + step / 6, -ypos - 2 * step],
            [zpos - step / 6, -ypos - 2 * step],
            [zpos, -ypos],
        ]

        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=self.color))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=self.color))

        ax.add_patch(
            mpatches.Circle(xy=(zpos, ypos + 2 * step), radius=radius, color=self.color)
        )
        ax.add_patch(
            mpatches.Circle(
                xy=(zpos, -ypos - 2 * step), radius=radius, color=self.color
            )
        )

    def bokeh_patch(self, position, bk_ax):
        """Disk element patch.

        Patch that will be used to draw the shaft element using Bokeh library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        hover : Bokeh HoverTool
            Bokeh HoverTool axes
        """
        zpos, ypos, step = position

        # bokeh plot - coordinates to plot disks elements
        z_upper = [zpos, zpos + step / 6, zpos - step / 6]
        y_upper = [ypos, ypos + 2 * step, ypos + 2 * step]

        z_lower = [zpos, zpos + step / 6, zpos - step / 6]
        y_lower = [-ypos, -ypos - 2 * step, -ypos - 2 * step]

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
                radius=[step / 6],
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
            color=self.color,
            legend_label="Disk",
            name="ub_disk",
        )
        bk_ax.patches(
            xs="z_l",
            ys="y_l",
            source=source,
            alpha=1,
            line_width=2,
            color=self.color,
            name="ub_disk",
        )
        bk_ax.circle(
            x="z_circle",
            y="yu_circle",
            radius="radius",
            source=source_c,
            fill_alpha=1,
            color=self.color,
            name="uc_disk",
        )
        bk_ax.circle(
            x="z_circle",
            y="yl_circle",
            radius="radius",
            source=source_c,
            fill_alpha=1,
            color=self.color,
            name="lc_disk",
        )

        hover = HoverTool(names=["uc_disk", "lc_disk", "ub_disk", "lb_disk"])
        hover.tooltips = [
            ("Disk Node", "@elnum"),
            ("Polar Moment of Inertia", "@IP"),
            ("Diametral Moment of Inertia", "@ID"),
            ("Disk mass", "@mass"),
            ("Tag", "@tag"),
        ]
        hover.mode = "mouse"

        return hover

    @classmethod
    @check_units
    def from_geometry(
        cls,
        n,
        material,
        width,
        i_d,
        o_d,
        tag=None,
        scale_factor=1.0,
        color=bokeh_colors[9],
    ):
        """Create a disk element from geometry properties.

        This class method will create a disk element from geometry data.

        Parameters
        ----------
        n : int
            Node in which the disk will be inserted.
        material: ross.Material
             Disk material.
        width : float, pint.Quantity
            The disk width.
        i_d : float, pint.Quantity
            Inner diameter.
        o_d : float, pint.Quantity
            Outer diameter.
        tag : str, optional
            A tag to name the element
            Default is None
        scale_factor: float, optional
            The scale factor is used to scale the disk drawing.
            Default is 1.
        color : str, optional
            A color to be used when the element is represented.
            Default is '#b2182b' (Cardinal).

        Attributes
        ----------
        m : float
            Mass of the disk element.
        Id : float
            Diametral moment of inertia.
        Ip : float
            Polar moment of inertia

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

        return cls(n, m, Id, Ip, tag, scale_factor, color)

    @classmethod
    def from_table(
        cls, file, sheet_name=0, tag=None, scale_factor=1, color=bokeh_colors[9]
    ):
        """Instantiate one or more disks using inputs from an Excel table.

        A header with the names of the columns is required. These names should
        match the names expected by the routine (usually the names of the
        parameters, but also similar ones). The program will read every row
        bellow the header until they end or it reaches a NaN.

        Parameters
        ----------
        file : str
            Path to the file containing the disk parameters.
        sheet_name : int or str, optional
            Position of the sheet in the file (starting from 0) or its name.
            If none is passed, it is assumed to be the first sheet in the file.

        Returns
        -------
        disk : list
            A list of disk objects.

        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/shaft_si.xls'
        >>> list_of_disks = DiskElement.from_table(file_path, sheet_name="More")
        >>> list_of_disks[0]
        DiskElement(Id=0.0, Ip=0.0, m=15.12, color='#b2182b', n=3, tag=None)
        """
        parameters = read_table_file(file, "disk", sheet_name=sheet_name)
        list_of_disks = []
        for i in range(0, len(parameters["n"])):
            list_of_disks.append(
                cls(
                    n=parameters["n"][i],
                    m=parameters["m"][i],
                    Id=float(parameters["Id"][i]),
                    Ip=float(parameters["Ip"][i]),
                    tag=tag,
                    scale_factor=scale_factor,
                    color=color,
                )
            )
        return list_of_disks


class DiskElement6DoF(DiskElement):
    """A disk element for 6 DoFs.

    This class will create a disk element with 6 DoF from input data of inertia and
    mass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, pint.Quantity
        Mass of the disk element.
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float, optional
        The scale factor is used to scale the disk drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#b2182b' (Cardinal).

    Examples
    --------
    >>> disk = DiskElement6DoF(n=0, m=32, Id=0.2, Ip=0.3)
    >>> disk.Ip
    0.3
    """

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        The numbering of the degrees of freedom for each node.

        Being the following their ordering for a node:

        x_0 - horizontal translation
        y_0 - vertical translation
        z_0 - axial translation
        alpha_0 - rotation around horizontal
        beta_0  - rotation around vertical
        theta_0 - torsion around axial

        >>> disk = disk_example_6dof()
        >>> disk.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2, 'alpha_0': 3, 'beta_0': 4, 'theta_0': 5}
        """
        return dict(x_0=0, y_0=1, z_0=2, alpha_0=3, beta_0=4, theta_0=5)

    def M(self):
        """Mass matrix for an instance of a 6 DoF disk element.

        This method will return the mass matrix for an instance of a disk
        element with 6DoFs.

        Returns
        -------
        M : np.ndarray
            Mass matrix for the 6DoFs disk element.

        Examples
        --------
        >>> disk = DiskElement6DoF(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.M().round(2)
        array([[32.59,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  , 32.59,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  , 32.59,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.18,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.18,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.33]])
        """
        m = self.m
        Id = self.Id
        Ip = self.Ip
        # fmt: off
        M = np.array([
            [m,  0,  0,  0,  0,  0],
            [0,  m,  0,  0,  0,  0],
            [0,  0,  m,  0,  0,  0],
            [0,  0,  0, Id,  0,  0],
            [0,  0,  0,  0, Id,  0],
            [0,  0,  0,  0,  0, Ip],
        ])
        # fmt: on
        return M

    def K(self):
        """Stiffness matrix for an instance of a 6 DoF disk element.

        This method will return the stiffness matrix for an instance of a disk
        element with 6DoFs.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> disk = disk_example_6dof()
        >>> disk.K().round(2)
        array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.33, 0.  , 0.  ]])
        """
        Ip = self.Ip
        # fmt: off
        K = np.array([
            [0, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 0],
            [0, 0, 0, Ip, 0, 0],
        ])
        # fmt: on
        return K

    def C(self):
        """Damping matrix for an instance of a 6 DoF disk element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> disk = disk_example_6dof()
        >>> disk.C()
        array([[0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
        """
        C = np.zeros((6, 6))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a 6 DoF disk element.

        This method will return the gyroscopic matrix for an instance of a disk
        element.

        Returns
        -------
        G : np.ndarray
            Gyroscopic matrix for the disk element.

        Examples
        --------
        >>> disk = DiskElement6DoF(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.G().round(2)
        array([[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.33,  0.  ],
               [ 0.  ,  0.  ,  0.  , -0.33,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
        """
        Ip = self.Ip
        # fmt: off
        G = np.array([
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0, Ip, 0],
            [0, 0, 0, -Ip,  0, 0],
            [0, 0, 0,   0,  0, 0],
        ])
        # fmt: on
        return G


def disk_example():
    """Create an example of disk element.

    This function returns an instance of a simple disk. The purpose is to make available
    a simple model so that doctest can be written using it.

    Returns
    -------
    disk : ross.DiskElement
        An instance of a disk object.

    Examples
    --------
    >>> disk = disk_example()
    >>> disk.Ip
    0.32956362
    """
    disk = DiskElement(0, 32.589_727_65, 0.178_089_28, 0.329_563_62)
    return disk


def disk_example_6dof():
    """Create an example of disk element.

    This function returns an instance of a simple disk. The purpose is to make available
    a simple model so that doctest can be written using it.

    Returns
    -------
    disk : ross.DiskElement6DoF
        An instance of a disk object.

    Examples
    --------
    >>> disk = disk_example_6dof()
    >>> disk.Ip
    0.32956362
    """
    disk = DiskElement6DoF(0, 32.589_727_65, 0.178_089_28, 0.329_563_62)
    return disk
