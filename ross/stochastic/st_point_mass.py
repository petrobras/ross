"""Point mass element module for STOCHASTIC ROSS.

This module creates an instance of random point mass for stochastic
analysis.
"""
from ross.point_mass import PointMass
from ross.stochastic.st_results_elements import plot_histogram
from ross.units import check_units

__all__ = ["ST_PointMass", "st_pointmass_example"]


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
    color : str, optional
        A color to be used when the element is represented.
        Default is "DarkSalmon".
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
    >>> len(list(iter(elms)))
    5
    """

    @check_units
    def __init__(
        self,
        n,
        m=None,
        mx=None,
        my=None,
        tag=None,
        color="DarkSalmon",
        is_random=None,
    ):
        attribute_dict = dict(
            n=n,
            m=m,
            mx=mx,
            my=my,
            tag=tag,
            color=color,
        )

        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random point mass elements.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> elm = srs.st_pointmass_example()
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
            ***check the correct type for each key in ST_PointMass
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
        attributes from ross.PointMass.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.PointMass class.
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
        >>> elm = srs.st_pointmass_example()
        >>> fig = elm.plot_random_var(["mx"])
        >>> # fig.show()
        """
        label = dict(
            mx="Mass on the X direction",
            my="Mass on the Y direction",
            m="Mass",
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


def st_pointmass_example():
    """Return an instance of a simple random point mass.

    The purpose is to make available a simple model so that doctest can be
    written using it.

    Returns
    -------
    elm : ross.stochastic.ST_PointMass
        An instance of a random point mass element object.

    Examples
    --------
    >>> import ross.stochastic as srs
    >>> elm = srs.st_pointmass_example()
    >>> len(list(iter(elm)))
    2
    """
    mx = [2.0, 2.5]
    my = [3.0, 3.5]
    elm = ST_PointMass(n=1, mx=mx, my=my, is_random=["mx", "my"])
    return elm
