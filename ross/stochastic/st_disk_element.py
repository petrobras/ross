import bokeh.palettes as bp
import numpy as np

from ross.disk_element import DiskElement
from ross.stochastic.st_materials import ST_Material

bokeh_colors = bp.RdGy[11]

__all__ = ["ST_DiskElement"]


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
        Default is '#b2182b' (Cardinal).
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
    >>> len(list(elms.__iter__()))
    5
    """

    def __init__(
        self, n, m, Id, Ip, tag=None, color=bokeh_colors[9], is_random=None,
    ):
        attribute_dict = dict(n=n, m=m, Id=Id, Ip=Ip, tag=tag, color=color)

        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random disk elements.
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
            ***check the correct type for each key in ST_ShaftElement
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
        attributes from DiskElement.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ShaftElement class.
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

    @classmethod
    def from_geometry(
        cls, n, material, width, i_d, o_d, tag=None, is_random=None,
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
        >>> len(list(elms.__iter__()))
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
            n=n, material=material, width=width, i_d=i_d, o_d=o_d, tag=tag,
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
