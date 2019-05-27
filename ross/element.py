from abc import ABC, abstractmethod

import pandas as pd
import toml


class Element(ABC):
    """Element class.
    This class is a general class to be called for other files which
    create specific elements for the user
    """

    def __init__(self):
        pass

    def save(self, file_name):
        pass

    @staticmethod
    def load(self, file_name):
        pass

    @abstractmethod
    def M(self):
        pass

    @abstractmethod
    def C(self):
        pass

    @abstractmethod
    def K(self):
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

