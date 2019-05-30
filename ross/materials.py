"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""
import os

import toml

import ross as rs

__all__ = ["Material", "steel"]


class Material:
    """Material.

    Class used to create a material and define its properties.
    Density and at least 2 arguments from E, G_s and Poisson
    should be provided.

    Run Material.available_materials() for materials already provided.

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

    def __init__(self, name, rho, **kwargs):

        assert name is not None, "Name not provided"
        assert type(name) is str, "Name must be a string"
        assert " " not in name, "Spaces are not allowed in Material name"
        assert (
            sum([1 if i in ["E", "G_s", "Poisson"] else 0 for i in kwargs]) > 1
        ), "At least 2 arguments from E, G_s and Poisson should be provided"

        self.name = name
        self.rho = rho
        self.E = kwargs.get("E", None)
        self.Poisson = kwargs.get("Poisson", None)
        self.G_s = kwargs.get("G_s", None)
        self.color = kwargs.get("color", "#525252")

        if self.E is None:
            self.E = self.G_s * (2 * (1 + self.Poisson))
        elif self.G_s is None:
            self.G_s = self.E / (2 * (1 + self.Poisson))
        elif self.Poisson is None:
            self.Poisson = (self.E / (2 * self.G_s)) - 1

    def __eq__(self, other):
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __repr__(self):
        selfE = "{:.2e}".format(self.E)
        selfPoisson = "{:.2e}".format(self.Poisson)
        selfrho = "{:.2e}".format(self.rho)
        selfGs = "{:.2e}".format(self.G_s)

        return (
            f"Material"
            f"(name={self.name}, rho={selfrho}, G_s={selfGs} "
            f"E={selfE}, Poisson={selfPoisson} color={self.color!r}"
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
        with open("available_materials.toml", "w") as f:
            toml.dump(data, f)

    @staticmethod
    def load_data():
        try:
            with open("available_materials.toml", "r") as f:
                data = toml.load(f)
        except FileNotFoundError:
            data = {"Materials": {}}
            Material.dump_data(data)
        return data

    @staticmethod
    def use_material(name):
        run_path = os.getcwd()
        ross_path = os.path.dirname(rs.__file__)
        os.chdir(ross_path)

        data = Material.load_data()
        try:
            material = data["Materials"][name]
            return Material(**material)
        except KeyError:
            raise KeyError("There isn't a instanced material with this name.")
        os.chdir(run_path)

    @staticmethod
    def remove_material(name):
        run_path = os.getcwd()
        ross_path = os.path.dirname(rs.__file__)
        os.chdir(ross_path)

        data = Material.load_data()
        try:
            del data["Materials"][name]
        except KeyError:
            return "There isn't a saved material with this name."
        Material.dump_data(data)
        os.chdir(run_path)

    @staticmethod
    def available_materials():
        run_path = os.getcwd()
        ross_path = os.path.dirname(rs.__file__)
        os.chdir(ross_path)

        try:
            data = Material.load_data()
            return list(data["Materials"].keys())
        except FileNotFoundError:
            return "There is no saved materials."
        os.chdir(run_path)

    def save_material(self):
        run_path = os.getcwd()
        ross_path = os.path.dirname(rs.__file__)
        os.chdir(ross_path)

        data = Material.load_data()
        data["Materials"][self.name] = self.__dict__
        Material.dump_data(data)


steel = Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
