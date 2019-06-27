from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd
import toml


class Element(ABC):
    """Element class.
    This class is a general class to be called for other files which
    create specific elements for the user
    """

    def __init__(self, n):
        self.n = n
        self.dof_mapping = None
        pass

    def save(self, file_name):
        pass

    @staticmethod
    def load(self, file_name):
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
        """Giroscopic matrix."""
        pass

    def summary(self):
        """A summary for the element.

        A pandas series with the element properties as variables.
        """
        attributes = self.__dict__
        attributes["type"] = self.__class__.__name__
        return pd.Series(attributes)

    @staticmethod
    def load_data(file_name):
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
        with open(file_name, "w") as f:
            toml.dump(data, f)

    @abstractmethod
    def dof_mapping(self):
        """Should return a dictionary with a mapping between degree of freedom
        and its index.

        Example considering a shaft element:
        def dof_mapping(self):
            return dict(
                x0=0,
                y0=1,
                alpha0=2,
                beta0=3,
                x1=4,
                y1=5,
                alpha1=6,
                beta1=7,
            )
        """
        pass

    def dof_local_index(self):
        """Get the local index for a element specific degree of freedom."""
        dof_mapping = self.dof_mapping()
        dof_tuple = namedtuple("LocalIndex", dof_mapping)
        global_index = dof_tuple(**dof_mapping)

        return global_index

    def dof_global_index(self):
        """Get the global index for a element specific degree of freedom."""
        dof_mapping = self.dof_mapping()
        dof_tuple = namedtuple("GlobalIndex", dof_mapping)

        for k, v in dof_mapping.items():
            dof_mapping[k] = 4 * self.n + v

        global_index = dof_tuple(**dof_mapping)

        return global_index
