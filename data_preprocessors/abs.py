import numpy as np
from abc import ABC, abstractmethod


class AbstractDataPreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    This class defines the basic structure for data preprocessing,
    including methods for loading data, preprocessing, and splitting
    the data into training, validation, and testing sets.

    Attributes
    ----------
    data_path : str
        Path to the data file.
    train_ratio : float, optional
        Ratio of training data. Default is 0.6.
    valid_ratio : float, optional
        Ratio of validation data. Default is 0.2.
    update_dataset_params : dict
        Dictionary to store parameters for updating the dataset params.
    update_model_params : dict
        Dictionary to store parameters for updating the model params.
    update_trainer_params : dict
        Dictionary to store parameters for updating the trainer params.

    Methods
    -------
    load_data()
        Abstract method to load data.
    preprocess()
        Abstract method to preprocess the data.
    split_data(preprocessed_data)
        Abstract method to split the preprocessed data into different sets.
    """

    @abstractmethod
    def __init__(self, data_path, *args, **kwargs):
        """
        Initialize the AbstractDataPreprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        train_ratio : float, optional
            Ratio of training data. Default is 0.6.
        valid_ratio : float, optional
            Ratio of validation data. Default is 0.2.
        """
        self.data_path = data_path
        self.train_ratio = kwargs.get("train_ratio", 0.6)
        self.valid_ratio = kwargs.get("valid_ratio", 0.2)
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}

    @abstractmethod
    def load_data(self):
        """
        Abstract method to load data.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Abstract method to preprocess the data.

        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        pass

    @abstractmethod
    def split_data(self, preprocessed_data):
        """
        Abstract method to split the preprocessed data.
        Parameters
        ----------
        preprocessed_data : Any
            The preprocessed data to be split. The type depends on the implementation.

        Returns
        -------
        tuple
            A tuple containing the training, validation, and test data sets.
        """
        pass
