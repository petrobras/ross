from ross.bearing_seal_element import BearingElement

__all__ = ["ST_BearingElement"]


class ST_BearingElement:
    """Random bearing element.

    Creates an object containing a list with random instances of
    BearingElement.

    Considering constant coefficients, use an 1-D array to make it random.
    Considering varying coefficients to the frequency, use and 2-D array to
    make it random (*see the Examples below).

    Parameters
    ----------
    n: int
        Node which the bearing will be located in
    kxx: float, 1-D array, 2-D array
        Direct stiffness in the x direction.
    cxx: float, 1-D array, 2-D array
        Direct damping in the x direction.
    kyy: float, 1-D array, 2-D array, optional
        Direct stiffness in the y direction.
        (defaults to kxx)
    kxy: float, 1-D array, 2-D array, optional
        Cross coupled stiffness in the x direction.
        (defaults to 0)
    kyx: float, 1-D array, 2-D array, optional
        Cross coupled stiffness in the y direction.
        (defaults to 0)
    cyy: float, 1-D array, 2-D array, optional
        Direct damping in the y direction.
        (defaults to cxx)
    cxy: float, 1-D array, 2-D array, optional
        Cross coupled damping in the x direction.
        (defaults to 0)
    cyx: float, 1-D array, 2-D array, optional
        Cross coupled damping in the y direction.
        (defaults to 0)
    frequency: array, optional
        Array with the frequencies (rad/s).
    tag: str, optional
        A tag to name the element
        Default is None.
    n_link: int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor: float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    is_random : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"]

    Attributes
    ----------
    elements_list : list
        display the list with random bearing elements.

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs

    # Uncertanties on constant bearing coefficients

    >>> s = 10
    >>> kxx = np.random.uniform(1e6, 2e6, s)
    >>> cxx = np.random.uniform(1e6, 2e6, s)
    >>> elms = srs.ST_BearingElement(n=1,
    ...                              kxx=kxx,
    ...                              cxx=cxx,
    ...                              is_random = ["kxx", "cxx"],
    ...                              )
    >>> len(list(elms.__iter__()))
    10

    # Uncertanties on bearing coefficients varying with frequency

    >>> s = 5
    >>> kxx = [np.random.uniform(1e6, 2e6, s),
    ...        np.random.uniform(2.3e6, 3.3e6, s)]
    >>> cxx = [np.random.uniform(1e3, 2e3, s),
    ...        np.random.uniform(2.1e3, 3.1e3, s)]
    >>> frequency = np.linspace(500, 800, len(kxx))
    >>> elms = srs.ST_BearingElement(n=1,
    ...                              kxx=kxx,
    ...                              cxx=cxx,
    ...                              frequency=frequency,
    ...                              is_random = ["kxx", "cxx"],
    ...                              )
    >>> len(list(elms.__iter__()))
    5
    """

    def __init__(
        self,
        n,
        kxx,
        cxx,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        frequency=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        is_random=None,
    ):

        if "frequency" in is_random:
            raise ValueError("frequency can not be a random variable")

        if kyy is None:
            kyy = kxx
            if "kxx" in is_random and "kyy" not in is_random:
                is_random.append("kyy")
        if cyy is None:
            cyy = cxx
            if "cxx" in is_random and "cyy" not in is_random:
                is_random.append("cyy")

        attribute_dict = dict(
            n=n,
            kxx=kxx,
            cxx=cxx,
            kyy=kyy,
            kxy=kxy,
            kyx=kyx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
            frequency=frequency,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
        )
        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random bearing elements.
        """
        return iter(self.random_var(self.is_random, self.attribute_dict))

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from BearingElement.

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

        try:
            for i in range(len(args_dict[is_random[0]][0])):
                arg = []
                for key, value in args_dict.items():
                    if key in is_random:
                        arg.append([value[j][i] for j in range(len(value))])
                    else:
                        arg.append(value)
                new_args.append(arg)

        except TypeError:
            for i in range(len(args_dict[is_random[0]])):
                arg = []
                for key, value in args_dict.items():
                    if key in is_random:
                        arg.append(value[i])
                    else:
                        arg.append(value)
                new_args.append(arg)

        f_list = (BearingElement(*arg) for arg in new_args)

        return f_list
