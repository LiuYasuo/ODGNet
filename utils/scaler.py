from abc import ABC, abstractmethod
import numpy as np


class Scaler(ABC):
    def __init__(self, axis=None):
        self.axis = axis

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass


class StandardScaler(Scaler):
    def __init__(self, axis=0, mean=None, std=None, *args, **kwargs):
        super(StandardScaler, self).__init__(axis)
        self.mean = None
        self.std = None

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.mean = np.mean(data, axis=self.axis)
        self.std = np.std(data, axis=self.axis)
        self.std[self.std == 0] = 1  # avoid dividing by 0

    def transform(self, data, index=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if index is None:
            return (data - self.mean) / self.std
        else:
            return (data - self.mean[index]) / self.std[index]

    def inverse_transform(self, data, index=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if index is None:
            return data * self.std + self.mean
        else:
            return data * self.std[index] + self.mean[index]


class MinMaxScaler(Scaler):
    def __init__(self, axis=0, min=None, max=None, *args, **kwargs):
        super(MinMaxScaler, self).__init__(axis)
        self.min = None
        self.max = None

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.min = np.min(data, axis=self.axis)
        self.max = np.max(data, axis=self.axis)

    def transform(self, data, index=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if index is None:
            return (data - self.min) / (self.max - self.min)
        else:
            return (data - self.min[index]) / (self.max[index] - self.min[index])

    def inverse_transform(self, data, index=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if index is None:
            return data * (self.max - self.min) + self.min
        else:
            return data * (self.max[index] - self.min[index]) + self.min[index]
