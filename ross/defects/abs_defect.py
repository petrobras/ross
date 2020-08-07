from abc import ABC, abstractmethod

__all__ = [
    "Defect",
]


class Defect(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def force(self):
        pass
