from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class AbstractDataset(Dataset, ABC):
    """
    An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    `__getitem__`, and `__len__` methods.
    """

    def __init__(self, data, *args, **kwargs):
        """
        Initialize the AbstractDataset.
        """
        self.data = data

    @abstractmethod
    def __getitem__(self, index):
        """
        Abstract method to retrieve an item from the dataset.
        Returns
        -------
        Any
            The item at the specified index.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Abstract method to get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        pass
