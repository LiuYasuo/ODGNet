import numpy as np
import pandas as pd

from data_preprocessors.abs import AbstractDataPreprocessor


class ElectricityDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for Electricity dataset.
    """

    def __init__(
        self,
        data_path,
        adj_mx_path=None,
        steps_per_day=24,  # interval is 1 hour
        add_feature_time_of_day=True,
        add_feature_day_of_week=True,
        add_feature_day_of_month=True,
        add_feature_day_of_year=True,
        *args,
        **kwargs,
    ):
        """
        Initialize the AirQualityDataPreprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file. .xlsx file
        adj_mx_path : str, optional
            Path to the adjacency matrix. Default is None. Note this dataset does not have an original adjacency matrix.
        steps_per_day : int, optional
            Number of steps per day. Default is 24.
        add_feature_time_of_day : bool, optional
            Whether to add the time of day feature. Default is True.
        add_feature_day_of_week : bool, optional
            Whether to add the day of week feature. Default is True.
        add_feature_day_of_month : bool, optional
            Whether to add the day of month feature. Default is True.
        add_feature_day_of_year : bool, optional
            Whether to add the day of year feature. Default is True.

        """

        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.datetime_index = None
        self.adj_mx = None
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.adj_mx_path = adj_mx_path if adj_mx_path is not None else None
        self.steps_per_day = steps_per_day
        self.add_feature_time_of_day = add_feature_time_of_day
        self.add_feature_day_of_week = add_feature_day_of_week
        self.add_feature_day_of_month = add_feature_day_of_month
        self.add_feature_day_of_year = add_feature_day_of_year
        self.load_data()


    def load_data(self):
        """
        Load data from the specified file path.
        """
        df = pd.read_csv(self.data_path)
        df_index = pd.to_datetime(df["date"].values, format="%Y-%m-%d %H:%M:%S")
        df = df[df.columns[1:]]
        data = np.expand_dims(df.values, axis=-1)
        self.data = data
        self.datetime_index = df_index

    def preprocess(self):
        """
        Preprocess the loaded data.

        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        data = self.data
        datetime_index = self.datetime_index
        l, n, f = data.shape

        # add temporal feature
        feature_list = [data]
        if self.add_feature_time_of_day:
            # numerical time_of_day
            tod = [
                i % self.steps_per_day / self.steps_per_day
                for i in range(data.shape[0])
            ]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(tod_tiled)

        if self.add_feature_day_of_week:
            # numerical day_of_week
            dow = datetime_index.dayofweek / 7
            dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)

        if self.add_feature_day_of_month:
            # numerical day_of_month
            dom = (
                datetime_index.day - 1
            ) / 31  # df.index.day starts from 1. We need to minus 1 to make it start from 0.
            dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(dom_tiled)

        if self.add_feature_day_of_year:
            # numerical day_of_year
            doy = (
                datetime_index.dayofyear - 1
            ) / 366  # df.index.month starts from 1. We need to minus 1 to make it start from 0.
            doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(doy_tiled)

        preprocessed_data = np.concatenate(feature_list, axis=-1)
        if self.adj_mx_path is not None:
            self.adj_mx = np.load(self.adj_mx_path)
            self.update_trainer_params["adj_mx"] = self.adj_mx
        self.update_model_params["adj_mx"] = None  # [self.asym_adj(R_adj), self.asym_adj(np.transpose(R_adj))]
        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["out_dim"] = self.forecast_len
        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["forecast_len"] = self.forecast_len
        return preprocessed_data

    def split_data(self, preprocessed_data):
        """
        Split the preprocessed data into training, validation, and testing sets.

        Parameters
        ----------
        preprocessed_data : np.ndarray
            The preprocessed data array to be split.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing the training, validation, and test data arrays, respectively.
        """
        total_length = len(preprocessed_data)
        train_end = int(self.train_ratio * total_length)
        valid_end = int((self.train_ratio + self.valid_ratio) * total_length)

        train_data = preprocessed_data[:train_end]
        valid_data = preprocessed_data[train_end- self.history_len:valid_end]
        test_data = preprocessed_data[valid_end- self.history_len:]

        return train_data, valid_data, test_data


# test
if __name__ == "__main__":
    data_path = "data/regression_dataset/Electricity/Electricity.csv"
    preprocessor = ElectricityDataPreprocessor(data_path)
    preprocessed_data = preprocessor.preprocess()
    train_data, valid_data, test_data = preprocessor.split_data(preprocessed_data)
    print(f"train_data shape: {train_data.shape}")
    print(f"valid_data shape: {valid_data.shape}")
    print(f"test_data shape: {test_data.shape}")
