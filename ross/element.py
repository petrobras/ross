from inspect import signature
from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd
import re

from ross.utils import load_data, dump_data


class Element(ABC):
    """Element class.

    This class is a general abstract class to be implemented in other files, in order to
    create specific elements for the user.
    """

    def __init__(self, n, tag=None):
        self.n = n
        self.tag = tag

    def save(self, file):
        """Save the element in a .toml or .json file.

        This function will save the element to a .toml or .json file.
        The file will have all the argument's names and values that are needed to
        reinstantiate the element.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the element will be saved in.
            The format is determined by the file extension (.toml or .json).

        Examples
        --------
        >>> # Example using DiskElement
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> from ross.disk_element import disk_example
        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'disk.toml'
        >>> disk = disk_example()
        >>> disk.save(file)
        """
        # get __init__ arguments
        args_list = list(signature(self.__init__).parameters)
        args = {arg: getattr(self, arg) for arg in args_list}
        try:
            data = load_data(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}_{self.tag}"] = args
        dump_data(data, file)

    @classmethod
    def read_toml_data(cls, data):
        """Read and parse data stored in a .toml or .json file.

        The data passed to this method needs to be according to the
        format saved by the .save() method.

        Parameters
        ----------
        data : dict
            Dictionary obtained from toml.load() or json.load().

        Returns
        -------
        The element object.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> from ross.bearing_seal_element import bearing_example
        >>> from ross.bearing_seal_element import BearingElement
        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'bearing1.toml'
        >>> bearing1 = bearing_example()
        >>> bearing1.save(file)
        >>> bearing1_loaded = BearingElement.load(file)
        >>> bearing1 == bearing1_loaded
        True
        """
        args_list = set(signature(cls.__init__).parameters).intersection(data.keys())
        required_data = {arg: data[arg] for arg in args_list}

        return cls(**required_data)

    @classmethod
    def load(cls, file):
        data = load_data(file)
        # extract single dictionary in the data
        data = list(data.values())[0]
        return cls.read_toml_data(data)

    @classmethod
    def get_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.get_subclasses())
        return subclasses

    @abstractmethod
    def M(self):
        """Mass matrix.

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.M(0)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        pass

    @abstractmethod
    def C(self, frequency):
        """Frequency dependent damping coefficients matrix.

        Parameters
        ----------
        frequency: float
            The frequency in which the coefficients depend on.

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.C(0)
        array([[200.,   0.,   0.],
               [  0., 150.,   0.],
               [  0.,   0.,  50.]])
        """
        pass

    @abstractmethod
    def K(self, frequency):
        """Frequency dependent stiffness coefficients matrix.

        Parameters
        ----------
        frequency: float
            The frequency in which the coefficients depend on.

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.K(0)
        array([[1000000.,       0.,       0.],
               [      0.,  800000.,       0.],
               [      0.,       0.,  100000.]])
        """
        pass

    @abstractmethod
    def G(self):
        """Gyroscopic matrix.

        Returns
        -------
        A matrix of floats.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.G()
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        pass

    def summary(self):
        """Present a summary for the element.

        A pandas series with the element properties as variables.

        Returns
        -------
        A pandas series.

        Examples
        --------
        >>> # Example using DiskElement
        >>> from ross.disk_element import disk_example
        >>> disk = disk_example()
        >>> disk.summary() # doctest: +ELLIPSIS
        n                             0
        n_l                           0
        n_r                           0...
        """
        attributes = self.__dict__
        attributes["type"] = self.__class__.__name__
        return pd.Series(attributes)

    @abstractmethod
    def dof_mapping(self):
        """Degrees of freedom mapping.

        Should return a dictionary with a mapping between degree of freedom
        and its index.

        Returns
        -------
        dof_mapping: dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2}
        """
        pass

    def dof_local_index(self):
        """Get the local index for a element specific degree of freedom.

        Returns
        -------
        local_index: namedtupple
            A named tuple containing the local index.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.dof_local_index()
        LocalIndex(x_0=0, y_0=1, z_0=2)
        """
        dof_mapping = self.dof_mapping()
        dof_tuple = namedtuple("LocalIndex", dof_mapping)
        local_index = dof_tuple(**dof_mapping)

        return local_index

    def get_class_name_prefix(self, index=None):
        """Extract prefix of the class name preceding 'Element',
        insert spaces before uppercase letters, and append an index
        number at the end.

        Parameters
        ----------
        index : int, optional
            The index number to append at the end of the resulting string.
            Default is None.

        Returns
        -------
        prefix : str
            The processed class name prefix.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.get_class_name_prefix()
        'Bearing'
        """
        class_name = self.__class__.__name__

        if "Shaft" in class_name:
            prefix = "Shaft Element"
        else:
            prefix = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name.split("Element")[0])

        if index is None:
            return prefix
        else:
            return f"{prefix} {index}"
