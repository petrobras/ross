import inspect
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path

import pandas as pd
import toml


class Element(ABC):
    """Element class.

    This class is a general abstract class to be implemented in other files, in order to
    create specific elements for the user.
    """

    def __init__(self, n, tag=None):
        self.n = n
        self.tag = tag

    def save(self, file):
        """Save the element in a .toml file.

        This function will save the element to a .toml file.
        The file will have all the argument's names and values that are needed to
        reinstantiate the element.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the element will be saved in.

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
        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}_{self.tag}"] = args
        with open(file, "w") as f:
            toml.dump(data, f)

    @classmethod
    def read_toml_data(cls, data):
        """Read and parse data stored in a .toml file.

        The data passed to this method needs to be according to the
        format saved in the .toml file by the .save() method.

        Parameters
        ----------
        data : dict
            Dictionary obtained from toml.load().

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
        return cls(**data)

    @classmethod
    def load(cls, file):
        data = toml.load(file)
        # extract single dictionary in the data
        data = list(data.values())[0]
        return cls.read_toml_data(data)

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
        >>> bearing.M()
        array([[0., 0.],
               [0., 0.]])
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
        array([[200.,   0.],
               [  0., 150.]])
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
        array([[1000000.,       0.],
               [      0.,  800000.]])
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
        array([[0., 0.],
               [0., 0.]])
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
        {'x_0': 0, 'y_0': 1}
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
        LocalIndex(x_0=0, y_0=1)
        """
        dof_mapping = self.dof_mapping()
        dof_tuple = namedtuple("LocalIndex", dof_mapping)
        local_index = dof_tuple(**dof_mapping)

        return local_index
