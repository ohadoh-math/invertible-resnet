"""
The interface a design implements.
"""


import abc
from torch.utils.data import Dataset
from torch.nn import Module


class Design(abc.ABC):
    """
    Basic design facilities.
    """

    @abc.abstractmethod
    def get_dataset(self) -> Dataset:
        """
        Get the current dataset that loads train data from the design.
        """
        ...

    @abc.abstractmethod
    def update_design(self, model: Module):
        """
        Update the design with the new version of the trained model
        """
        ...
