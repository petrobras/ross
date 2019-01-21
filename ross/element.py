from abc import ABC, abstractmethod


class Element(ABC):
    """Element class.
    This class is a general class to be called for other files which
    create specific elements for the user
    """

    def __init__(self):
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
