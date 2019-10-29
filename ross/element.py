from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd
import toml


class Element(ABC):
    """Element class.
    This class is a general abstract class to be implemented in other files, in order to
    create specific elements for the user.
    """

    def __init__(self, n):
        self.n = n
        self.dof_mapping = None
        pass

    def save(self, file_name):
        """Saves the element in a file.
        Parameters
        ----------
        file_name: string
            The name of the file the element will be saved in.
        Returns
        -------
        None
        Examples
        --------
        >>> # Example using DiskElement
        >>> from ross.disk_element import disk_example
        >>> disk = disk_example()
        >>> disk.save('DiskElement.toml')
        """
        pass

    @staticmethod
    def load(file_name):
        """Loads elements saved in a file.
        Parameters
        ----------
        file_name: string
            The name of the file to be loaded.
        Returns
        -------
        The element object.
        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> from ross.bearing_seal_element import BearingElement
        >>> bearing1 = bearing_example()
        >>> bearing1.save('BearingElement.toml')
        >>> list_of_bearings = BearingElement.load('BearingElement.toml')
        >>> bearing1 == list_of_bearings[0]
        True
        """
        pass

    @abstractmethod
    def M(self):
        """Mass matrix.
        Parameters
        ----------

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
        Parameters
        ----------

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
        """A summary for the element.

        A pandas series with the element properties as variables.

        Parameters
        ----------

        Returns
        -------
        A pandas series.

        Examples
        --------
        >>> # Example using DiskElement
        >>> from ross.disk_element import disk_example
        >>> disk = disk_example()
        >>> disk.summary()
        n                  0
        n_l                0
        n_r                0
        m            32.5897
        Id          0.178089
        Ip          0.329564
        tag             None
        color        #b2182b
        type     DiskElement
        dtype: object
        """
        attributes = self.__dict__
        attributes["type"] = self.__class__.__name__
        return pd.Series(attributes)

    @staticmethod
    def load_data(file_name):
        """Loads elements data saved in a toml format.

        Parameters
        ----------
        file_name: string
            The name of the file to be loaded.

        Returns
        -------
        data: dict
            The element parameters presented as a dictionary.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> from ross.bearing_seal_element import BearingElement
        >>> bearing = bearing_example()
        >>> bearing.save('BearingElement.toml')
        >>> BearingElement.load_data('BearingElement.toml') # doctest: +ELLIPSIS
        {'BearingElement': {'0': {'n': 0, 'kxx': [1000000.0, 1000000.0,...
        """
        try:
            with open(file_name, "r") as f:
                data = toml.load(f)
                if data == {"": {}}:
                    data = {file_name[:-5]: {}}

        except FileNotFoundError:
            data = {file_name[:-5]: {}}
            Element.dump_data(data, file_name)
        return data

    @staticmethod
    def dump_data(data, file_name):
        """Dumps element data in a toml file.

        Parameters
        ----------
        data: dict
            The data that should be dumped.
        file_name: string
            The name of the file the data will be dumped in.

        Returns
        -------

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> from ross.bearing_seal_element import BearingElement
        >>> bearing = bearing_example()
        >>> bearing.save('BearingElement.toml')
        >>> data = BearingElement.load_data('BearingElement.toml')
        >>> BearingElement.dump_data(data, 'bearing_data.toml')
        """
        with open(file_name, "w") as f:
            toml.dump(data, f)

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

    def dof_global_index(self):
        """Get the global index for a element specific degree of freedom.

        Returns
        -------
        global_index: namedtupple
            A named tuple containing the global index.

        Examples
        --------
        >>> # Example using BearingElement
        >>> from ross.bearing_seal_element import bearing_example
        >>> bearing = bearing_example()
        >>> bearing.dof_global_index()
        GlobalIndex(x_0=0, y_0=1)
        """
        dof_mapping = self.dof_mapping()
        global_dof_mapping = {}
        for k, v in dof_mapping.items():
            dof_letter, dof_number = k.split("_")
            global_dof_mapping[dof_letter + "_" + str(int(dof_number) + self.n)] = v
        dof_tuple = namedtuple("GlobalIndex", global_dof_mapping)

        for k, v in global_dof_mapping.items():
            global_dof_mapping[k] = 4 * self.n + v

        global_index = dof_tuple(**global_dof_mapping)

        return global_index
