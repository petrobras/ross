"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""

from pathlib import Path

import numpy as np
import toml

from .units import check_units
from ross.units import Q_

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
        Density (kg/m**3).
    E : float, pint.Quantity, optional
        Young's modulus (N/m**2).
    G_s : float, pint.Quantity, optional
        Shear modulus (N/m**2).
    Poisson : float, optional
        Poisson ratio (dimensionless).
    specific_heat : float, optional
        Specific heat (J/(kg*K)).
    thermal_conductivity : float, optional
        Thermal conductivity (W/(m*K)).
    color : str, optional
        Color that will be used on plots.

    Examples
    --------
    >>> from ross.units import Q_
    >>> AISI4140 = Material(name="AISI4140", rho=7850, E=203.2e9, G_s=80e9)
    >>> Steel = Material(name="Steel", rho=Q_(7.81, 'g/cm**3'), E=211e9, G_s=81.2e9)
    >>> AISI4140.Poisson
    0.27
    >>> Steel.rho
    7809.999999999999
    """

    @check_units
    def __init__(
        self,
        name,
        rho,
        E=None,
        G_s=None,
        Poisson=None,
        specific_heat=None,
        thermal_conductivity=None,
        color="#525252",
        **kwargs,
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
        if E is None:
            E = G_s * (2 * (1 + Poisson))
        elif G_s is None:
            G_s = E / (2 * (1 + Poisson))
        elif Poisson is None:
            Poisson = (E / (2 * G_s)) - 1

        self.rho = float(rho)
        self.E = float(E)
        self.G_s = float(G_s)
        self.Poisson = float(Poisson)
        self.specific_heat = float(specific_heat) if specific_heat is not None else 0.0
        self.thermal_conductivity = (
            float(thermal_conductivity) if thermal_conductivity is not None else 0.0
        )
        self.color = color

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
        >>> import ross as rs
        >>> steel = rs.Material.load_material('Steel')
        >>> AISI4140 = rs.Material.load_material('AISI4140')
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
        """Return a string representation of a material.

        Returns
        -------
        A string representation of a material object.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.load_material('Steel')
        >>> steel # doctest: +ELLIPSIS
        Material(name="Steel", rho=7.81000e+03, G_s=8.12000e+10, E=2.11000e+11, specific_heat = 4.34000e+02, thermal_conductivity = 6.05000e+01,color='#525252')
        """
        selfE = "{:.5e}".format(self.E)
        selfrho = "{:.5e}".format(self.rho)
        selfGs = "{:.5e}".format(self.G_s)
        selfspecific_heat = "{:.5e}".format(self.specific_heat)
        selfthermal_conductivity = "{:.5e}".format(self.thermal_conductivity)

        return (
            f"Material"
            f'(name="{self.name}", rho={selfrho}, G_s={selfGs}, '
            f"E={selfE}, specific_heat = {selfspecific_heat}, thermal_conductivity = {selfthermal_conductivity},"
            f"color={self.color!r})"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings

        Examples
        --------
        >>> import ross as rs
        >>> print(rs.Material.load_material('Steel'))
        Steel
        -----------------------------------
        Density         (kg/m**3): 7810.0
        Young`s modulus (N/m**2):  2.11e+11
        Shear modulus   (N/m**2):  8.12e+10
        Poisson coefficient     :  0.29926108
        Specific heat   (J/(kg*K)): 434.0
        Thermal conductivity (W/(m*K)): 60.5
        """
        return (
            f"{self.name}"
            f'\n{35*"-"}'
            f"\nDensity         (kg/m**3): {self.rho:{2}.{8}}"
            f"\nYoung`s modulus (N/m**2):  {self.E:{2}.{8}}"
            f"\nShear modulus   (N/m**2):  {self.G_s:{2}.{8}}"
            f"\nPoisson coefficient     :  {self.Poisson:{2}.{8}}"
            f"\nSpecific heat   (J/(kg*K)): {self.specific_heat:{2}.{8}}"
            f"\nThermal conductivity (W/(m*K)): {self.thermal_conductivity:{2}.{8}}"
        )

    @staticmethod
    def dump_data(data):
        """Save material properties.

        This is an auxiliary function to save the materials properties in the save
        method.

        Parameters
        ----------
        data : dict
            Dictionary containing all data needed to instantiate the Object.
        """
        with open(AVAILABLE_MATERIALS_PATH, "w") as f:
            toml.dump(data, f)

    @staticmethod
    def get_data():
        """Load material properties.

        This is an auxiliary function to load all saved materials properties in the
        load_material method.

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
    def load_material(name):
        """Load a material that is available in the data file.

        Returns
        -------
        ross.Material
            An object with the material properties.

        Raises
        ------
        KeyError
            Error raised if argument name does not match any material name in the file.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.load_material('Steel')
        """
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
        """Delete a saved ross.Material.

        Parameters
        ----------
        name : string
            Name of Material Object to be deleted.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.load_material('Steel')
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
        """Return a list of all saved material's name.

        Returns
        -------
        available_materials : list
            A list containing all saved material's names.

        Examples
        --------
        >>> import ross as rs
        >>> steel = rs.Material.load_material('Steel')
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
        """Save the material in the available_materials list."""
        data = Material.get_data()
        data["Materials"][self.name] = self.__dict__
        Material.dump_data(data)


steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
