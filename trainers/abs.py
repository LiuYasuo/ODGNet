import json
import os
from shutil import copyfile

import torch
from abc import ABC, abstractmethod

from utils.early_stop import EarlyStop
import utils.metrics as metrics_module


class AbstractTrainer(ABC):
    """
    Abstract base class for training machine learning models.

    Methods
    -------
    train(train_data_loader, eval_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs)
        Train the model with given data loaders and metrics.
    test(test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs)
        Test the model using the test data loader and specified metrics.
    save_checkpoint(filename="checkpoint.pth")
        Save the current training state as a checkpoint.
    load_checkpoint(filename="checkpoint.pth")
        Load training state from a checkpoint.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        *args,
        **kwargs,
    ):
        """
        Initialize the trainer.
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer used for training the model.
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler.
        scaler : Scaler
            Scaler object for data normalization.
        model_save_path : str
            Path to save the trained model_state, and checkpoint.
        result_save_dir_path : str
            Directory path to save training results and logs.
        max_epoch_num : int
            Maximum number of epochs to train the model.
        enable_early_stop : bool, optional
            Flag to enable early stopping mechanism. Default is False.
        early_stop_patience : int, optional
            Number of epochs to wait for improvement before stopping. Default is 5.
        early_stop_min_is_best : bool, optional
            Flag to determine if lower values indicate better performance. Default is True.

        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir_path = os.path.dirname(model_save_path)
        self.result_save_dir_path = result_save_dir_path
        self.epoch_now = 0
        self.max_epoch_num = max_epoch_num
        self.enable_early_stop = enable_early_stop
        if self.enable_early_stop:
            self.early_stop = EarlyStop(early_stop_patience, early_stop_min_is_best)
        else:
            self.early_stop = None
        self.device = next(self.model.parameters()).device
        self.min_loss = torch.finfo(torch.float32).max

    def train(
        self,
        train_data_loader,
        eval_data_loader,
        metrics=("mae", "rmse", "mape"),
        *args,
        **kwargs,
    ):
        """
        Train the model using the provided training and evaluation data loaders.

        Parameters
        ----------
        train_data_loader : DataLoader
            DataLoader for the training data.
        eval_data_loader : DataLoader
            DataLoader for the evaluation data.
        metrics : tuple of str
            Metrics to evaluate the model performance. Default is ("mae", "rmse", "mape").

        Returns
        -------
        dict
            A dictionary containing training and evaluation results for each epoch.
        """
        tmp_state_save_path = os.path.join(self.model_save_dir_path, "temp.pkl")
        epoch_result_list = []

        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader)
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
            )

            # check early stop
            if self.early_stop is not None and self.early_stop.reach_stop_criteria(
                eval_loss
            ):
                self.early_stop.reset()
                break

            # save best model
            if eval_loss < self.min_loss:
                self.min_loss = eval_loss
                torch.save(self.model.state_dict(), tmp_state_save_path)

            # lr scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)
        return epoch_result_json

    def test(self, test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs):
        """
        Test the model using the provided test data loader.
        Parameters
        ----------
        test_data_loader : DataLoader
            DataLoader for the test data.
        metrics : tuple of str
            Metrics to evaluate the model performance. Default is ("mae", "rmse", "mape").
        Returns
        -------
        tuple
            test_result : dict
                A dictionary containing the test results.
            y_pred
                Predicted values.
            y_true
                True values.

        """
        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        # evaluate on test set
        test_loss, metrics_evals, y_pred, y_true = self.evaluate(
            test_data_loader, metrics
        )
        test_result = {"loss": test_loss}
        for metric_name, metric_eval in metrics_evals:
            test_result[metric_name] = metric_eval
        return test_result, y_pred, y_true

    def save_checkpoint(self, filename="checkpoint.pth"):
        """
        Save the current training state as a checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "epoch": self.epoch_now,
            "min_loss": self.min_loss,
        }
        torch.save(checkpoint, os.path.join(self.model_save_dir_path, filename))

    def load_checkpoint(self, filename="checkpoint.pth"):
        """
        Load training state from a checkpoint.
        """

        filepath = os.path.join(self.model_save_dir_path, filename)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch_now = checkpoint.get("epoch", 1)
            self.min_loss = checkpoint.get("min_loss", torch.finfo(torch.float32).max)
        else:
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    def _save_epoch_result(self, epoch_result_list):
        """
        Save the loss and metrics for each epoch to a json file.
        """
        # save loss and metrics for each epoch to self.result_save_dir/epoch_result.json
        epoch_result = {}
        for epoch, train_loss, eval_loss, metrics_evals in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "eval_loss": eval_loss}
            for metric_name, metric_eval in metrics_evals:
                epoch_result[epoch][metric_name] = metric_eval
        with open(
            os.path.join(self.result_save_dir_path, "epoch_result.json"), "w"
        ) as f:
            json.dump(epoch_result, f, indent=4)
        return epoch_result

    @abstractmethod
    def loss_func(self, y_pred, y_true, *args, **kwargs):
        """
        Abstract method for computing the loss.

        To be implemented by subclasses.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        """
        Abstract method for training the model for one epoch.

        To be implemented by subclasses.
        Returns
        -------
        torch.Tensor
            Train loss.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader, metrics, *args, **kwargs):
        """
        Abstract method for evaluating the model.

        To be implemented by subclasses.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader for the evaluation data.
        metrics : tuple of str
            Metrics to evaluate the model performance.

        Returns
        -------
        tuple
            loss : torch.Tensor
                Evaluation loss.
            eval_results : list
                List of computed metric values.
            y_pred
                Predicted values.
            y_true
                True values.
        """
        pass

    @staticmethod
    def get_eval_result(y_pred, y_true, metrics=("mae", "rmse", "mape")):
        """
        Compute evaluation metrics for the given predictions and true values.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.
        metrics : tuple of str
            Metrics to be computed.

        Returns
        -------
        list
            List of computed metric values.
        """
        eval_results = []
        for metric_name in metrics:
            eval_func_name = "get_{}".format(metric_name)
            eval_func = getattr(metrics_module, eval_func_name, None)

            if eval_func is None:
                raise AttributeError(
                    f"Function '{eval_func_name}' not found in 'utils.metrics'."
                )

            result = eval_func(y_pred, y_true)
            eval_results.append(result)

        return eval_results
