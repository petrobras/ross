"""Disk element module for STOCHASTIC ROSS.

This module creates an instance of random disk element for stochastic
analysis.
"""
import numpy as np

from ross.disk_element import DiskElement
from ross.stochastic.st_materials import ST_Material
from ross.stochastic.st_results_elements import plot_histogram
from ross.units import check_units

__all__ = ["ST_DiskElement", "st_disk_example"]


class ST_DiskElement:
    """Random disk element.

    Creates an object containing a list with random instances of DiskElement.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, list
        Mass of the disk element.
        Input a list to make it random.
    Id : float, list
        Diametral moment of inertia.
        Input a list to make it random.
    Ip : float, list
        Polar moment of inertia
        Input a list to make it random.
    tag : str, optional
        A tag to name the element
        Default is None
    color : str, optional
        A color to be used when the element is represented.
        Default is "Firebrick".
    is_random : list
        List of the object attributes to become random.
        Possibilities:
            ["m", "Id", "Ip"]

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs
    >>> elms = srs.ST_DiskElement(n=1,
    ...                           m=30.0,
    ...                           Id=np.random.uniform(0.20, 0.40, 5),
    ...                           Ip=np.random.uniform(0.15, 0.25, 5),
    ...                           is_random=["Id", "Ip"],
    ...                           )
    >>> len(list(iter(elms)))
    5
    """

    @check_units
    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        tag=None,
        color="Firebrick",
        is_random=None,
    ):
        attribute_dict = dict(n=n, m=m, Id=Id, Ip=Ip, tag=tag, color=color)

        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random disk elements.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> elm = srs.st_disk_example()
        >>> len(list(iter(elm)))
        2
        """
        return iter(self.random_var(self.is_random, self.attribute_dict))

    def __getitem__(self, key):
        """Return the value for a given key from attribute_dict.

        Parameters
        ----------
        key : str
            A class parameter as string.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Returns
        -------
        Return the value for the given key.

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> elms = srs.ST_DiskElement(n=1,
        ...                           m=30.0,
        ...                           Id=np.random.uniform(0.20, 0.40, 5),
        ...                           Ip=np.random.uniform(0.15, 0.25, 5),
        ...                           is_random=["Id", "Ip"],
        ...                           )
        >>> elms["m"]
        30.0
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))

        return self.attribute_dict[key]

    def __setitem__(self, key, value):
        """Set new parameter values for the object.

        Function to change a parameter value.
        It's not allowed to add new parameters to the object.

        Parameters
        ----------
        key : str
            A class parameter as string.
        value : The corresponding value for the attrbiute_dict's key.
            ***check the correct type for each key in ST_DiskElement
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> elms = srs.ST_DiskElement(n=1,
        ...                           m=30.0,
        ...                           Id=np.random.uniform(0.20, 0.40, 5),
        ...                           Ip=np.random.uniform(0.15, 0.25, 5),
        ...                           is_random=["Id", "Ip"],
        ...                           )
        >>> elms["Id"] = np.linspace(0.1, 0.3, 5)
        >>> elms["Id"]
        array([0.1 , 0.15, 0.2 , 0.25, 0.3 ])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from ross.DiskElement.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.DiskElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : generator
            Generator of random objects.
        """
        args_dict = args[0]
        new_args = []
        for i in range(len(args_dict[is_random[0]])):
            arg = []
            for key, value in args_dict.items():
                if key in is_random:
                    arg.append(value[i])
                else:
                    arg.append(value)
            new_args.append(arg)
        f_list = (DiskElement(*arg) for arg in new_args)

        return f_list

    def plot_random_var(self, var_list=None, histogram_kwargs=None, plot_kwargs=None):
        """Plot histogram and the PDF.

        This function creates a histogram to display the random variable
        distribution.

        Parameters
        ----------
        var_list : list, optional
            List of random variables, in string format, to plot.
            Default is plotting all the random variables.
        histogram_kwargs : dict, optional
            Additional key word arguments can be passed to change
            the plotly.go.histogram (e.g. histnorm="probability density", nbinsx=20...).
            *See Plotly API to more information.
        plot_kwargs : dict, optional
            Additional key word arguments can be passed to change the plotly go.figure
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...).
            *See Plotly API to more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            A figure with the histogram plots.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> elm = srs.st_disk_example()
        >>> fig = elm.plot_random_var(["m"])
        >>> # fig.show()
        """
        label = dict(
            m="Mass",
            Id="Diametral moment of inertia",
            Ip="Polar moment of inertia",
        )
        if var_list is None:
            var_list = self.is_random
        elif not all(var in self.is_random for var in var_list):
            raise ValueError(
                "Random variable not in var_list. Select variables from {}".format(
                    self.is_random
                )
            )

        return plot_histogram(
            self.attribute_dict, label, var_list, histogram_kwargs={}, plot_kwargs={}
        )

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
        is_random=None,
    ):
        """Random disk element.

        Creates an object containing a list with random instances of
        DiskElement.from_geometry.

        Parameters
        ----------
        n: int
            Node in which the disk will be inserted.
        material: ross.Material, list of ross.Material
            Disk material.
            Input a list to make it random.
        width: float, list
            The disk width.
            Input a list to make it random.
        i_d: float, list
            Inner diameter.
            Input a list to make it random.
        o_d: float, list
            Outer diameter.
            Input a list to make it random.
        tag : str, optional
            A tag to name the element
            Default is None
        is_random : list
            List of the object attributes to become random.
            Possibilities:
                ["material", "width", "i_d", "o_d"]

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> from ross.materials import steel
        >>> i_d=np.random.uniform(0.05, 0.06, 5)
        >>> o_d=np.random.uniform(0.35, 0.39, 5)
        >>> elms = srs.ST_DiskElement.from_geometry(n=1,
        ...                                         material=steel,
        ...                                         width=0.07,
        ...                                         i_d=i_d,
        ...                                         o_d=o_d,
        ...                                         is_random=["i_d", "o_d"],
        ...                                         )
        >>> len(list(iter(elms)))
        5
        """
        if isinstance(material, ST_Material):
            material = list(material.__iter__())
            rho = np.array([m.rho for m in material])
        else:
            rho = material.rho

        if type(width) == list:
            width = np.array(width)
        if type(i_d) == list:
            i_d = np.array(i_d)
        if type(o_d) == list:
            o_d = np.array(o_d)

        attribute_dict = dict(
            n=n,
            material=material,
            width=width,
            i_d=i_d,
            o_d=o_d,
            tag=tag,
        )
        size = len(attribute_dict[is_random[0]])

        for k, v in attribute_dict.items():
            if k not in is_random:
                v = np.full(size, v)
            else:
                v = np.array(v)

        m = 0.25 * rho * np.pi * width * (o_d ** 2 - i_d ** 2)
        # fmt: off
        Id = (
            0.015625 * rho * np.pi * width * (o_d ** 4 - i_d ** 4)
            + m * (width ** 2) / 12
        )
        # fmt: on
        Ip = 0.03125 * rho * np.pi * width * (o_d ** 4 - i_d ** 4)

        is_random = ["m", "Id", "Ip"]

        return cls(n, m, Id, Ip, tag, is_random=is_random)


def st_disk_example():
    """Return an instance of a simple random disk.

    The purpose is to make available a simple model so that doctest can be
    written using it.

    Returns
    -------
    elm : ross.stochastic.ST_DiskElement
        An instance of a random disk element object.

    Examples
    --------
    >>> import ross.stochastic as srs
    >>> elm = srs.st_disk_example()
    >>> len(list(iter(elm)))
    2
    """
    elm = ST_DiskElement(
        n=1,
        m=[30, 40],
        Id=[0.2, 0.3],
        Ip=[0.5, 0.7],
        is_random=["m", "Id", "Ip"],
    )
    return elm
