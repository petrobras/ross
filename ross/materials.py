"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""
import os
import numpy as np
import toml
from pathlib import Path


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
    E : float
        Young's modulus (N/m**2).
    G_s : float
        Shear modulus (N/m**2).
    rho : float
        Density (N/m**3).
    color : str
        Can be used on plots.

    Examples
    --------
    >>> AISI4140 = Material(name='AISI4140', rho=7850, E=203.2e9, G_s=80e9)
    >>> Steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    >>> AISI4140.Poisson
    0.27

    """

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
                "At least 2 arguments from E, G_s" "and Poisson should be provided "
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
        """Material is considered equal if properties are equal."""
        self_list = [v for v in self.__dict__.values() if isinstance(v, (float, int))]
        other_list = [v for v in self.__dict__.values() if isinstance(v, (float, int))]

        if np.allclose(self_list, other_list):
            return True
        else:
            return False

    def __repr__(self):
        selfE = "{:.3e}".format(self.E)
        selfPoisson = "{:.3e}".format(self.Poisson)
        selfrho = "{:.3e}".format(self.rho)
        selfGs = "{:.3e}".format(self.G_s)

        return (
            f"Material"
            f'(name="{self.name}", rho={selfrho}, G_s={selfGs}, '
            f"E={selfE}, Poisson={selfPoisson}, color={self.color!r})"
        )

    def __str__(self):
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
        with open(AVAILABLE_MATERIALS_PATH, "w") as f:
            toml.dump(data, f)

    @staticmethod
    def load_data():
        """Loads materials from data file."""
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
        data = Material.load_data()
        try:
            material = data["Materials"][name]
            return Material(**material)
        except KeyError:
            raise KeyError("There isn't a instanced material with this name.")

    @staticmethod
    def remove_material(name):
        data = Material.load_data()
        try:
            del data["Materials"][name]
        except KeyError:
            return "There isn't a saved material with this name."
        Material.dump_data(data)

    @staticmethod
    def available_materials():

        try:
            data = Material.load_data()
            return list(data["Materials"].keys())
        except FileNotFoundError:
            return "There is no saved materials."
        os.chdir(run_path)

    def save_material(self):
        """Saves the material in the available_materials list."""
        data = Material.load_data()
        data["Materials"][self.name] = self.__dict__
        Material.dump_data(data)


steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
