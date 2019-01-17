import pandas as pd
from abc import ABC

class Element(ABC):
    """Element class.
    This class is a general class to be called for other files which
    create specific elements for the user
    """

    def __init__(self):
        pass

    def summary(self):
        """A summary for the element.
        A pandas series with the element properties as variables.
        """
        attributes = self.__dict__
        attributes["type"] = self.__class__.__name__
        return pd.Series(attributes)
    
    @ABC.abstractmethod    
    def M(self):
        pass
    
    @ABC.abstractmethod    
    def C(self):
        pass
    
    @ABC.abstractmethod    
    def K(self):
        pass
