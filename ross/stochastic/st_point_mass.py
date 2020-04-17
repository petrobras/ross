from ross.point_mass import PointMass


class ST_PointMass:
    """Random point mass element.

    Creates an object containing a list with random instances of PointMass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m: float, list, optional
        Mass for the element.
        Input a list to make it random.
    mx: float, list optional
        Mass for the element on the x direction.
        Input a list to make it random.
    my: float, optional
        Mass for the element on the y direction.
        Input a list to make it random.
    tag : str, optional
        A tag to name the element
        Default is None
    is_random : list
        List of the object attributes to become random.
        Possibilities:
            ["m", "mx", "my"]

    Attributes
    ----------
    elements : list
        display the list with random point mass elements.

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs
    >>> elms = srs.ST_PointMass(n=1,
    ...                         mx=np.random.uniform(2.0, 2.5, 5),
    ...                         my=np.random.uniform(2.0, 2.5, 5),
    ...                         is_random=["mx", "my"],
    ...                         )
    >>> len(list(elms.__iter__()))
    5
    """

    def __init__(
        self, n, m=None, mx=None, my=None, tag=None, is_random=None,
    ):
        attribute_dict = dict(n=n, m=m, mx=mx, my=my, tag=tag,)

        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random point mass elements.
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
        >>> elms = srs.ST_PointMass(n=1,
        ...                         mx=np.random.uniform(2.0, 2.5, 5),
        ...                         my=np.random.uniform(2.0, 2.5, 5),
        ...                         is_random=["mx", "my"],
        ...                         )
        >>> elms["n"]
        1
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
        >>> elms = srs.ST_PointMass(n=1,
        ...                         mx=np.random.uniform(2.0, 2.5, 5),
        ...                         my=np.random.uniform(2.0, 2.5, 5),
        ...                         is_random=["mx", "my"],
        ...                         )
        >>> elms["mx"] = np.linspace(1.0, 2.0, 5)
        >>> elms["mx"]
        array([1.  , 1.25, 1.5 , 1.75, 2.  ])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from PointMass.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the PointMass class.
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
        f_list = (PointMass(*arg) for arg in new_args)

        return f_list
