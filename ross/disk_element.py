"""Disk Element module.

This module defines the DiskElement classes which will be used to represent equipments
attached to the rotor shaft, which add mainly mass and inertia to the system.
"""

import numpy as np
from plotly import graph_objects as go

from ross.element import Element
from ross.units import check_units
from ross.utils import read_table_file

__all__ = ["DiskElement"]


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
    scale_factor: float or str, optional
        The scale factor is used to scale the disk drawing.
        For disks it is also possible to provide 'mass' as the scale factor.
        In this case the code will calculate scale factors for each disk based
        on the disk with the higher mass. Notice that in this case you have to
        create all disks with the scale_factor='mass'.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Firebrick'.

    Examples
    --------
    >>> disk = DiskElement(n=0, m=32, Id=0.2, Ip=0.3)
    >>> disk.Ip
    0.3
    """

    @check_units
    def __init__(self, n, m, Id, Ip, tag=None, scale_factor=1.0, color="Firebrick"):
        self.n = int(n)
        self.n_l = n
        self.n_r = n

        self.m = float(m)
        self.Id = float(Id)
        self.Ip = float(Ip)
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
        if "DiskElement" in other.__class__.__name__:
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

                except KeyError:
                    false_number += 1
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
            f"n={self.n}, scale_factor={self.scale_factor}, tag={self.tag!r})"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings

        Example
        -------
        >>> print(DiskElement(n=0, m=32, Id=0.223, Ip=0.31223, tag="Disk"))
        Tag:                      Disk
        Node:                     0
        Mass           (kg):      32.0
        Diam. inertia  (kg*m**2): 0.223
        Polar. inertia (kg*m**2): 0.31223
        """
        return (
            f"Tag:                      {self.tag}"
            f"\nNode:                     {self.n}"
            f"\nMass           (kg):      {self.m:{2}.{5}}"
            f"\nDiam. inertia  (kg*m**2): {self.Id:{2}.{5}}"
            f"\nPolar. inertia (kg*m**2): {self.Ip:{2}.{5}}"
        )

    def __hash__(self):
        return hash(self.tag)

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

        >>> disk = disk_example()
        >>> disk.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2, 'alpha_0': 3, 'beta_0': 4, 'theta_0': 5}
        """
        return dict(x_0=0, y_0=1, z_0=2, alpha_0=3, beta_0=4, theta_0=5)

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
        >>> disk.K().round(2)
        array([[0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
        """
        K = np.zeros((6, 6))

        return K

    def Kdt(self):
        """Dynamic stiffness matrix for an instance of a disk element.

        Stiffness matrix for the disk element associated with the
        transient motion. It needs to be multiplied by the angular
        acceleration when considered in time dependent analyses.

        Returns
        -------
        Kdt : np.ndarray
            A matrix of floats containing the values of the dynamic
            stiffness matrix.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.Kdt().round(2)
        array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.33, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])
        """
        Ip = self.Ip
        # fmt: off
        Kdt = np.array([[0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0, Ip, 0, 0],
                        [0, 0, 0,  0, 0, 0]])
        # fmt: on
        return Kdt

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

    def _patch(self, position, fig):
        """Disk element patch.

        Patch that will be used to draw the shaft element using plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        zpos, ypos, yc_pos, scale_factor = position
        radius = scale_factor / 8

        # coordinates to plot disks elements
        z_upper = [zpos, zpos + scale_factor / 25, zpos - scale_factor / 25, zpos]
        y_upper = [ypos, ypos + 2 * scale_factor, ypos + 2 * scale_factor, ypos]

        z_lower = [zpos, zpos + scale_factor / 25, zpos - scale_factor / 25, zpos]
        y_lower = [-ypos, -ypos - 2 * scale_factor, -ypos - 2 * scale_factor, -ypos]

        z_pos = z_upper
        z_pos.append(None)
        z_pos.extend(z_lower)

        y_pos = y_upper
        y_upper.append(None)
        y_pos.extend(y_lower)

        customdata = [self.n, self.Ip, self.Id, self.m]
        hovertemplate = (
            f"Disk Node: {customdata[0]}<br>"
            + f"Polar Inertia: {customdata[1]:.3e}<br>"
            + f"Diametral Inertia: {customdata[2]:.3e}<br>"
            + f"Disk mass: {customdata[3]:.3f}<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=z_pos,
                y=[y + yc_pos if y is not None else None for y in y_pos],
                customdata=[customdata] * len(z_pos),
                text=hovertemplate,
                mode="lines",
                fill="toself",
                fillcolor=self.color,
                opacity=0.8,
                line=dict(width=2.0, color=self.color),
                showlegend=False,
                name=self.tag,
                legendgroup="disks",
                hoveron="points+fills",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=y_upper[1] - radius + yc_pos,
                x1=zpos + radius,
                y1=y_upper[1] + radius + yc_pos,
                fillcolor=self.color,
                line_color=self.color,
            )
        )
        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=y_lower[1] - radius + yc_pos,
                x1=zpos + radius,
                y1=y_lower[1] + radius + yc_pos,
                fillcolor=self.color,
                line_color=self.color,
            )
        )

        return fig

    @staticmethod
    def calculate_mass(rho, width, i_d, o_d):
        return rho * np.pi * width * (o_d**2 - i_d**2) / 4

    @staticmethod
    def calculate_Ip(m, i_d, o_d):
        return m * (o_d**2 + i_d**2) / 8

    @staticmethod
    def calculate_Id(Ip, m, width):
        return 1 / 2 * Ip + 1 / 12 * m * width**2

    @staticmethod
    def calculate_width(rho, m, i_d, o_d):
        return 4 * m / (rho * np.pi * (o_d**2 - i_d**2))

    @classmethod
    @check_units
    def from_geometry(
        cls, n, material, width, i_d, o_d, tag=None, scale_factor=1.0, color="Firebrick"
    ):
        """Create a disk element from geometry properties.

        This class method will create a disk element from geometry data.
        Properties are calculated as per :cite:`friswell2010dynamics`, appendix 1
        for a hollow cylinder:

        Mass:

        :math:`m = \\rho \\pi w (d_o^2 - d_i^2) / 4`

        Polar moment of inertia:

        :math:`I_p = m (d_o^2 + d_i^2) / 8`

        Diametral moment of inertia:

        :math:`I_d = \\frac{1}{2} I_p + \\frac{1}{12} m w^2`

        Where :math:`\\rho` is the material density, :math:`w` is the disk width,
        :math:`d_o` is the outer diameter and :math:`d_i` is the inner diameter.

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
            Default is 'Firebrick' (Cardinal).

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
        m = cls.calculate_mass(material.rho, width, i_d, o_d)
        Ip = cls.calculate_Ip(m, i_d, o_d)
        Id = cls.calculate_Id(Ip, m, width)

        return cls(n, m, Id, Ip, tag, scale_factor, color)

    @classmethod
    def from_table(cls, file, sheet_name=0, tag=None, scale_factor=None, color=None):
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
        tag_list : list, optional
            list of tags for the disk elements.
            Default is None
        scale_factor: list, optional
            List of scale factors for the disk elements patches.
            The scale factor is used to scale the disk drawing.
            Default is 1.
        color : list, optional
            A color to be used when the element is represented.
            Default is 'Firebrick'.

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
        DiskElement(Id=0.0, Ip=0.0, m=15.12, color='Firebrick', n=3, scale_factor=1, tag=None)
        """
        parameters = read_table_file(file, "disk", sheet_name=sheet_name)
        if tag is None:
            tag = [None] * len(parameters["n"])
        if scale_factor is None:
            scale_factor = [1] * len(parameters["n"])
        if color is None:
            color = ["Firebrick"] * len(parameters["n"])

        list_of_disks = []
        for i in range(0, len(parameters["n"])):
            list_of_disks.append(
                cls(
                    n=parameters["n"][i],
                    m=parameters["m"][i],
                    Id=float(parameters["Id"][i]),
                    Ip=float(parameters["Ip"][i]),
                    tag=tag[i],
                    scale_factor=scale_factor[i],
                    color=color[i],
                )
            )
        return list_of_disks


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
