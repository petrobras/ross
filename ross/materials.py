"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""
import numpy as np
import toml
from pathlib import Path
from .units import check_units


__all__ = ["Material", "steel"]

ROSS_PATH = Path(__file__).parent
AVAILABLE_MATERIALS_PATH = ROSS_PATH / "available_materials.toml"


class Material:
    """Material used on shaft and disks.

    Class used to create a material and define its properties.
    Density and at least 2 arguments from E, G_s and Poisson should be
    provided.

    You can run rs.Material.available_materials() to get a list of materials
    already provided.

    Parameters
    ----------
    name : str
        Material name.
    rho : float, pint.Quantity
        Density (N/m**3).
    E : float, pint.Quantity
        Young's modulus (N/m**2).
    G_s : float,
        Shear modulus (N/m**2).
    Poisson : float
        Poisson ratio (dimensionless).
    color : str
        Can be used on plots.

    Examples
    --------
    >>> AISI4140 = Material(name="AISI4140", rho=7850, E=203.2e9, G_s=80e9)
    >>> Steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    >>> AISI4140.Poisson
    0.27
    """

    @check_units
    def __init__(
        self, name, rho, E=None, G_s=None, Poisson=None, color="#525252", **kwargs
    ):
        self.name = str(name)
        if " " in name:
            raise ValueError("Spaces are not allowed in Material name")

        given_args = []
        for arg in ["E", "G_s", "Poisson"]:
            if locals()[arg] is not None:
                given_args.append(arg)
        if len(given_args) != 2:
            raise ValueError(
                "Exactly 2 arguments from E, G_s and Poisson should be provided"
            )
        self.name = name
        self.rho = rho
        self.E = E
        self.G_s = G_s
        self.Poisson = Poisson
        self.color = color

        if self.E is None:
            self.E = self.G_s * (2 * (1 + self.Poisson))
        elif self.G_s is None:
            self.G_s = self.E / (2 * (1 + self.Poisson))
        elif self.Poisson is None:
            self.Poisson = (self.E / (2 * self.G_s)) - 1

    def __eq__(self, other):
        """Function used to compare two Materials.
        Parameters
        ----------

        self : Material

        other: Material

        Returns
        -------

        bool
            True if all the Materials properties are equivalent.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.use_material('Steel')
        >>> AISI4140 = rs.Material.use_material('AISI4140')
        >>> steel == AISI4140
        False
        """
        self_list = [v for v in self.__dict__.values() if isinstance(v, (float, int))]
        other_list = [v for v in other.__dict__.values() if isinstance(v, (float, int))]

        if np.allclose(self_list, other_list):
            return True
        else:
            return False

    def __repr__(self):
        """Function used to give a representation of a Material element, when called.

        Parameters
        ----------

        self : Material

        Returns
        -------

        string : Representation of the given rs.Material.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.use_material('Steel')
        >>> steel # doctest: +ELLIPSIS
        Material(name="Steel", rho=7.81000e+03, G_s=8.12000e+10, E=2.11000e+11, color='#525252')
        """
        selfE = "{:.5e}".format(self.E)
        selfrho = "{:.5e}".format(self.rho)
        selfGs = "{:.5e}".format(self.G_s)

        return (
            f"Material"
            f'(name="{self.name}", rho={selfrho}, G_s={selfGs}, '
            f"E={selfE}, color={self.color!r})"
        )

    def __str__(self):
        """Function used to set what is shown when a Material is printed.

        Parameters
        ----------

        self : Material

        Returns
        -------

        str
            Containing all the Materials properties organized in a table.

        Examples
        --------
        >>> import ross as rs
        >>> print(rs.Material.use_material('Steel'))
        Steel
        -----------------------------------
        Density         (N/m**3): 7810.0
        Young`s modulus (N/m**2): 2.11e+11
        Shear modulus   (N/m**2): 8.12e+10
        Poisson coefficient     : 0.29926108
        """
        return (
            f"{self.name}"
            f'\n{35*"-"}'
            f"\nDensity         (N/m**3): {float(self.rho):{2}.{8}}"
            f"\nYoung`s modulus (N/m**2): {float(self.E):{2}.{8}}"
            f"\nShear modulus   (N/m**2): {float(self.G_s):{2}.{8}}"
            f"\nPoisson coefficient     : {float(self.Poisson):{2}.{8}}"
        )

    @staticmethod
    def dump_data(data):
        """Auxiliary function to save the materials properties in the save method.

        Parameters
        ----------

        data : dict
            Dictionary containing all data needed to instantiate the Object.

        Returns
        -------
        """
        with open(AVAILABLE_MATERIALS_PATH, "w") as f:
            toml.dump(data, f)

    @staticmethod
    def get_data():

        """Auxiliary function to load all saved materials properties in the use_material method.

        Parameters
        ----------

        Returns
        -------

        data : dict
            Containing all data needed to instantiate a Material Object.

        """
        try:
            with open(AVAILABLE_MATERIALS_PATH, "r") as f:
                data = toml.load(f)
        except FileNotFoundError:
            data = {"Materials": {}}
            Material.dump_data(data)
        return data

    @staticmethod
    def use_material(name):
        """Use material that is available in the data file."""
        data = Material.get_data()
        try:
            # Remove Poisson from dict and create material from E and G_s
            data["Materials"][name].pop("Poisson")
            material = data["Materials"][name]
            return Material(**material)
        except KeyError:
            raise KeyError("There isn't a instanced material with this name.")

    @staticmethod
    def remove_material(name):
        """Function used to delete a saved rs.Material.

        Parameters
        ----------

        name : string
            Name of Material Object to be deleted.

        Returns
        -------

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.use_material('Steel')
        >>> steel.name = 'test_material'
        >>> steel.save_material()
        >>> steel.remove_material('test_material')
        """
        data = Material.get_data()

        try:
            del data["Materials"][name]
        except KeyError:
            return "There isn't a saved material with this name."
        Material.dump_data(data)

    @staticmethod
    def available_materials():
        """Function used to get a list of all saved material's names.

        Parameters
        ----------

        Returns
        -------
        available_materials : list
            A list containing all saved material's names.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.use_material('Steel')
        >>> steel.name = 'test_material'
        >>> steel.save_material()
        >>> steel.remove_material('test_material')
        """

        try:
            data = Material.get_data()
            return list(data["Materials"].keys())
        except FileNotFoundError:
            return "There is no saved materials."

    def save_material(self):
        """Saves the material in the available_materials list."""
        data = Material.get_data()
        data["Materials"][self.name] = self.__dict__
        Material.dump_data(data)


steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
