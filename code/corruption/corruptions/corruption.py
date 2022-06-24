from abc import ABC, abstractmethod

class Corruption(ABC):
    """
    Abstract class for corrupting a given text.
    """

    @abstractmethod
    def __call__(self, text):
        """
        Corrupts the given text.
        """
        pass