"""Bearing element module for STOCHASTIC ROSS.

This module creates an instance of random bearing for stochastic analysis.
"""
# fmt: off
import numpy as np

from ross.bearing_seal_element import BearingElement
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_damping_matrix, calculate_stiffness_matrix)

# fmt: on

__all__ = ["ST_BearingElement"]


class ST_BearingElement:
    """Random bearing element.

    Creates an object containing a list with random instances of
    BearingElement.

    Considering constant coefficients, use an 1-D array to make it random.
    Considering varying coefficients to the frequency, use a 2-D array to
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
    >>> cxx = np.random.uniform(1e3, 2e3, s)
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
        >>> s = 5
        >>> kxx = np.random.uniform(1e6, 2e6, s)
        >>> cxx = np.random.uniform(1e3, 2e3, s)
        >>> elms = srs.ST_BearingElement(n=1,
        ...                              kxx=kxx,
        ...                              cxx=cxx,
        ...                              is_random = ["kxx", "cxx"],
        ...                              )
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
        >>> s = 5
        >>> kxx = np.random.uniform(1e6, 2e6, s)
        >>> cxx = np.random.uniform(1e3, 2e3, s)
        >>> elms = srs.ST_BearingElement(n=1,
        ...                              kxx=kxx,
        ...                              cxx=cxx,
        ...                              is_random = ["kxx", "cxx"],
        ...                              )
        >>> elms["kxx"] = np.linspace(3e6, 5e6, 5)
        >>> elms["kxx"]
        array([3000000., 3500000., 4000000., 4500000., 5000000.])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

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

    @classmethod
    def from_fluid_flow(
        cls,
        n,
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=None,
        load=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        is_random=None,
    ):
        """Instantiate a bearing using inputs from its fluid flow.

        Parameters
        ----------
        n : int
            The node in which the bearing will be located in the rotor.
        is_random : list
            List of the object attributes to become random.
            Possibilities:
                ["length", "omega", "p_in", "p_out", "radius_rotor",
                 "radius_stator", "visc", "rho", "eccentricity", "load"]
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

        Grid related
        ^^^^^^^^^^^^
        Describes the discretization of the problem
        nz: int
            Number of points along the Z direction (direction of flow).
        ntheta: int
            Number of points along the direction theta. NOTE: ntheta must be odd.
        nradius: int
            Number of points along the direction r.
        length: float, list
            Length in the Z direction (m).
            Input a list to make it random.

        Operation conditions
        ^^^^^^^^^^^^^^^^^^^^
        Describes the operation conditions.
        omega: float, list
            Rotation of the rotor (rad/s).
            Input a list to make it random.
        p_in: float, list
            Input Pressure (Pa).
            Input a list to make it random.
        p_out: float, list
            Output Pressure (Pa).
            Input a list to make it random.
        load: float, list
            Load applied to the rotor (N).
            Input a list to make it random.

        Geometric data of the problem
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Describes the geometric data of the problem.
        radius_rotor: float, list
            Rotor radius (m).
            Input a list to make it random.
        radius_stator: float, list
            Stator Radius (m).
            Input a list to make it random.
        eccentricity: float, list
            Eccentricity (m) is the euclidean distance between rotor and stator
            centers.
            The center of the stator is in position (0,0).
            Input a list to make it random.

        Fluid characteristics
        ^^^^^^^^^^^^^^^^^^^^^
        Describes the fluid characteristics.
        visc: float, list
            Viscosity (Pa.s).
            Input a list to make it random.
        rho: float, list
            Fluid density(Kg/m^3).
            Input a list to make it random.

        Returns
        -------
        random bearing: srs.ST_BearingElement
            A random bearing object.

        Examples
        --------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> nz = 30
        >>> ntheta = 20
        >>> nradius = 11
        >>> length = 0.03
        >>> omega = 157.1
        >>> p_in = 0.
        >>> p_out = 0.
        >>> radius_rotor = 0.0499
        >>> radius_stator = 0.05
        >>> eccentricity = (radius_stator - radius_rotor)*0.2663
        >>> visc = np.random.uniform(0.1, 0.2, 5)
        >>> rho = 860.0
        >>> elms = srs.ST_BearingElement.from_fluid_flow(
        ...     0, nz, ntheta, nradius, length,
        ...     omega, p_in, p_out, radius_rotor,
        ...     radius_stator, visc, rho,
        ...     eccentricity=eccentricity, is_random=["visc"]
        ... )
        >>> len(list(elms.__iter__()))
        5
        """
        attribute_dict = locals()
        size = len(attribute_dict[is_random[0]])
        args_dict = {
            "kxx": [],
            "kxy": [],
            "kyx": [],
            "kyy": [],
            "cxx": [],
            "cxy": [],
            "cyx": [],
            "cyy": [],
        }

        for k, v in attribute_dict.items():
            if k not in is_random:
                attribute_dict[k] = np.full(size, v)
            else:
                attribute_dict[k] = np.asarray(v)

        for i in range(size):
            fluid_flow = flow.FluidFlow(
                attribute_dict["nz"][i],
                attribute_dict["ntheta"][i],
                attribute_dict["nradius"][i],
                attribute_dict["length"][i],
                attribute_dict["omega"][i],
                attribute_dict["p_in"][i],
                attribute_dict["p_out"][i],
                attribute_dict["radius_rotor"][i],
                attribute_dict["radius_stator"][i],
                attribute_dict["visc"][i],
                attribute_dict["rho"][i],
                eccentricity=attribute_dict["eccentricity"][i],
                load=attribute_dict["load"][i],
            )
            c = calculate_damping_matrix(fluid_flow, force_type="short")
            k = calculate_stiffness_matrix(fluid_flow, force_type="short")
            args_dict["kxx"].append(k[0])
            args_dict["kxy"].append(k[1])
            args_dict["kyx"].append(k[2])
            args_dict["kyy"].append(k[3])
            args_dict["cxx"].append(c[0])
            args_dict["cxy"].append(c[1])
            args_dict["cyx"].append(c[2])
            args_dict["cyy"].append(c[3])

        return cls(
            n,
            kxx=args_dict["kxx"],
            cxx=args_dict["cxx"],
            kyy=args_dict["kyy"],
            kxy=args_dict["kxy"],
            kyx=args_dict["kyx"],
            cyy=args_dict["cyy"],
            cxy=args_dict["cxy"],
            cyx=args_dict["cyx"],
            frequency=[fluid_flow.omega],
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
            is_random=list(args_dict.keys()),
        )
