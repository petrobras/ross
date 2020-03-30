"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""
from collections.abc import Iterable

import numpy as np

from ross.materials import Material

__all__ = ["ST_Material"]


class ST_Material:
    """Create instance of Material with random parameters.

    Class used to create a material and define its properties.
    Density and at least 2 arguments from E, G_s and Poisson should be
    provided.

    If any material property is passed as iterable, the material becomes random.
    Inputing 1 or 2 arguments from E, G_s or Poisson as iterable will turn the
    third argument an iterable, but calculated based on the other two.

    For example:
        if E is iterable and G_s is float, then, Poisson is iterable and each
        term is calculated based on E values and G_s single value.

    You can run ross.Material.available_materials() to get a list of materials
    already provided.

    Parameters
    ----------
    name : str
        Material name.
    rho : float, list, pint.Quantity
        Density (kg/m**3).
        Input a list to make it random.
    E : float, list, pint.Quantity
        Young's modulus (N/m**2).
        Input a list to make it random.
    G_s : float, list
        Shear modulus (N/m**2).
        Input a list to make it random.
    Poisson : float, list
        Poisson ratio (dimensionless).
        Input a list to make it random.
    color : str
        Can be used on plots.

    Examples
    --------
    >>> # Steel with random Young's modulus.
    >>> import ross.stochastic as srs
    >>> E = np.random.uniform(208e9, 211e9, 5)
    >>> st_steel = srs.ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
    >>> len(list(st_steel.__iter__()))
    5
    """

    def __init__(
        self, name, rho, E=None, G_s=None, Poisson=None, color="#525252", **kwargs
    ):
        self.name = str(name)
        if " " in name:
            raise ValueError("Spaces are not allowed in Material name")

        i = 0
        for arg in ["E", "G_s", "Poisson"]:
            if locals()[arg] is not None:
                i += 1
        if i != 2:
            raise ValueError(
                "Exactly 2 arguments from E, G_s and Poisson should be provided"
            )

        is_random = []
        for par, _name in zip([rho, E, G_s, Poisson], ["rho", "E", "G_s", "Poisson"]):
            if isinstance(par, Iterable):
                par = np.array(par)
                is_random.append(_name)

        self.name = name
        self.rho = rho
        self.E = E
        self.G_s = G_s
        self.Poisson = Poisson
        self.color = color

        attribute_dict = dict(
            name=name, rho=rho, E=E, G_s=G_s, Poisson=Poisson, color=color,
        )
        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from ShaftElement.

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
        f_list = (Material(*arg) for arg in new_args)

        return f_list

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random material properties.
        """
        return iter(self.random_var(self.is_random, self.attribute_dict))
