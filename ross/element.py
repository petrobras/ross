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
        """Mass matrix."""
        pass

    @abstractmethod
    def C(self, frequency):
        """Frequency dependent damping coefficients matrix."""
        pass

    @abstractmethod
    def K(self, frequency):
        """Frequency dependent stiffness coefficients matrix."""
        pass

    @abstractmethod
    def G(self):
        """Gyroscopic matrix."""
        pass

    def summary(self):
        """A summary for the element.

        A pandas series with the element properties as variables.
        Returns
        -------
        A pandas series.
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
        dict
            The element parameters presented as a dictionary.
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
        """
        with open(file_name, "w") as f:
            toml.dump(data, f)

    @abstractmethod
    def dof_mapping(self):
        """Should return a dictionary with a mapping between degree of freedom
        and its index.

        Example considering a shaft element:
        def dof_mapping(self):
            return dict(
                x_0=0,
                y_0=1,
                alpha_0=2,
                beta_0=3,
                x_1=4,
                y_1=5,
                alpha_1=6,
                beta_1=7,
            )
        """
        pass

    def dof_local_index(self):
        """Get the local index for a element specific degree of freedom."""
        dof_mapping = self.dof_mapping()
        dof_tuple = namedtuple("LocalIndex", dof_mapping)
        local_index = dof_tuple(**dof_mapping)

        return local_index

    def dof_global_index(self):
        """Get the global index for a element specific degree of freedom."""
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
