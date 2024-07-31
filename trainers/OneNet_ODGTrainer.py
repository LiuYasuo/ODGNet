import json
import warnings
import os
from shutil import copyfile
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import tqdm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.metrics import metric, cumavg
from einops import rearrange

from trainers.abs import AbstractTrainer

import utils.metrics as metrics_module


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = torch.sigmoid(x)
        return x


class OneNet_ODGTrainer(AbstractTrainer):
    """
    A Trainer subclass for soft sensor data.
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
        forecast_len,
        inverse,
        online_learning,
        n_inner,
        lradj,
        learning_rate,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        input_index_list=None,
        output_index_list=None,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model for single step forecasting.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training the model.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler.
        scaler : Scaler
            Scaler object used for normalizing and denormalizing data.
        model_save_path : str
            Path to save the trained model.
        result_save_dir_path : str
            Directory path to save training and evaluation results.
        max_epoch_num : int
            The maximum number of epochs for training.
        enable_early_stop : bool, optional
            Whether to enable early stopping (default is False).
        early_stop_patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 5).
        early_stop_min_is_best : bool, optional
            Flag to indicate if lower values of loss indicate better performance (default is True).
        input_index_list : list, optional
            Indices of input variables in the dataset (default is None).
        output_index_list : list, optional
            Indices of output variables in the dataset (default is None).
        """
        super().__init__(
            model,
            optimizer,
            scheduler,
            scaler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            enable_early_stop,
            early_stop_patience,
            early_stop_min_is_best,
            input_index_list,
            output_index_list,
            *args,
            **kwargs,
        )
        self.forecast_len = forecast_len
        self.online = online_learning
        self.n_inner = n_inner
        self.early_stop_patience = early_stop_patience
        self.lradj = lradj
        self.learning_rate = learning_rate
        self.extlr = 0.001
        self.decision = MLP(n_inputs=forecast_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1,
                            act=nn.Tanh()).to(self.device)
        self.weight = torch.zeros(model.num_nodes, device=self.device)
        self.bias = torch.zeros(model.num_nodes, device=self.device)
        self.weight.requires_grad = True
        self.opt_w = optim.Adam([self.weight], lr=self.extlr)
        self.opt_bias = optim.Adam(self.decision.parameters(), lr=self.extlr)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.MSELoss()(y_pred, y_true)
        #loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0
        for i, (batch_x, batch_y, stamp) in enumerate(data_loader):
            self.optimizer.zero_grad()
            pred, true, loss_w, loss_bias = self._process_one_batch(batch_x, batch_y[:, :, :, 0], stamp)

            loss = self.loss_func(pred[0], true) + self.loss_func(pred[1], true)
            loss.backward()
            self.optimizer.step()
            self.model.store_grad()
            total_loss += loss.item()
        return total_loss / len(data_loader)


    def _process_one_batch(self, batch_x, batch_y, stamp, mode='train'):
        if mode == 'test' and self.online != 'none':
            return self._ol_one_batch(batch_x, batch_y, stamp)
        stamp = stamp.type(torch.LongTensor).to(self.device)
        x = batch_x.float().to(self.device)

        b, t, d = batch_y.shape
        loss1 = torch.sigmoid(self.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss1 = rearrange(loss1, 'b t d -> b (t d)')
        outputs, y1, y2 = self.model.forward_weight(x, stamp, loss1, 1 - loss1)

        batch_y = batch_y.float().to(self.device)

        loss_w = self.loss_func(outputs, rearrange(batch_y, 'b t d -> b (t d)'))
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()


        y1_w, y2_w = y1.view(b, t, d).detach(), y2.view(b, t, d).detach()
        true_w = batch_y.view(b, t, d).detach()
        loss1 = torch.sigmoid(self.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)

        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)

        self.bias = self.decision(inputs_decision.permute(0, 2, 1))
        weight = self.weight.view(1, 1, -1)
        weight = weight.repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = torch.sigmoid(weight + bias.repeat(1, t, 1))
        loss1 = rearrange(loss1, 'b t d -> b (t d)')
        loss2 = 1 - loss1

        y1_w = rearrange(y1_w, 'b t d -> b (t d)')
        y2_w = rearrange(y2_w, 'b t d -> b (t d)')
        true_w = rearrange(true_w, 'b t d -> b (t d)')

        loss_bias = self.loss_func(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        return [y1, y2], rearrange(batch_y, 'b t d -> b (t d)'), loss_w.detach().cpu().item(), loss_bias.detach().cpu().item()


    def _ol_one_batch(self, batch_x, batch_y, stamp):

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        stamp = stamp.type(torch.LongTensor).to(self.device)

        b, t, d = batch_y.shape
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        weight = self.weight.view(1, 1, -1)
        weight = weight.repeat(b, t, 1)

        bias = self.bias.view(-1, 1, d)
        loss1 = torch.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
        loss1 = rearrange(loss1, 'b t d -> b (t d)')

        outputs, y1, y2 = self.model.forward_weight(x, stamp, loss1, 1 - loss1)
        l1, l2 = self.loss_func(y1, true), self.loss_func(y2, true)
        loss = l1 + l2
        loss.backward()
        self.optimizer.step()
        self.model.store_grad()
        self.optimizer.zero_grad()

        y1_w, y2_w = y1.view(b, t, d).detach(), y2.view(b, t, d).detach()
        true_w = batch_y.view(b, t, d).detach()
        loss1 = torch.sigmoid(self.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)
        self.bias = self.decision(inputs_decision.permute(0, 2, 1))
        weight = self.weight.view(1, 1, -1)
        weight = weight.repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = torch.sigmoid(weight + bias.repeat(1, t, 1))
        loss1 = rearrange(loss1, 'b t d -> b (t d)')
        loss2 = 1 - loss1

        y1_w = rearrange(y1_w, 'b t d -> b (t d)')
        y2_w = rearrange(y2_w, 'b t d -> b (t d)')
        true_w = rearrange(true_w, 'b t d -> b (t d)')

        outputs_bias = loss1 * y1_w + loss2 * y2_w
        loss_bias = self.loss_func(outputs_bias, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        loss1 = torch.sigmoid(self.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss1 = rearrange(loss1, 'b t d -> b (t d)')
        loss_w = self.loss_func(loss1 * y1.detach() + (1 - loss1) * y2.detach(), rearrange(batch_y, 'b t d -> b (t d)'))
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()

        return outputs, rearrange(batch_y, 'b t d -> b (t d)')


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
        early_stopping = EarlyStopping(patience=self.early_stop_patience, verbose=True)
        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluateh
            eval_loss = self.vali(eval_data_loader)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss]
            )
            '''
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
            '''
            early_stopping(eval_loss, self.model, tmp_state_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.lradj, self.learning_rate)
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json


    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, stamp) in enumerate(vali_loader):
            pred, true, _, _ = self._process_one_batch(batch_x, batch_y[:, :, :, 0], stamp, mode='vali')
            pred = pred[0] * 0.5 + 0.5 * pred[1]
            loss = self.loss_func(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def inverse(self, data, mean, std):
        return data * std + mean


    def test(self, test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs):

        self.weight = torch.zeros(self.model.num_nodes, device = self.device)
        self.bias = torch.zeros(self.model.num_nodes, device = self.device)


        self.weight.requires_grad = True
        self.opt_w = optim.Adam([self.weight], lr=self.extlr)


        self.model.load_state_dict(torch.load(self.model_save_path)) #之前没写
        self.model.eval()
        for name, param in self.model.named_parameters():
            if 'nodevec1' in name:
                param.requires_grad = False
            if 'nodevec2' in name:
                param.requires_grad = False

        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        preds = []
        trues = []
        maes, mses, rmses, mapes = [], [], [], []
        for i, (batch_x, batch_y, stamp) in enumerate(tqdm(test_data_loader)):
            pred, true = self._process_one_batch(batch_x, batch_y[:, :, :, 0], stamp, mode='test')

            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)

        MAE, MSE, RMSE, MAPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes)
        mae, mse, rmse, mape = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1]
        print('mse:{}, mae:{}'.format(mse, mae))
        test_result = {}
        test_result['mae'] = mae
        test_result['mse'] = mse
        test_result['rmse'] = rmse
        test_result['mape'] = mape
        return test_result, preds, trues

    def _save_epoch_result(self, epoch_result_list):
        """
        Save the loss and metrics for each epoch to a json file.
        """
        # save loss and metrics for each epoch to self.result_save_dir/epoch_result.json
        epoch_result = {}
        for epoch, train_loss, eval_loss in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "eval_loss": eval_loss}
        with open(
            os.path.join(self.result_save_dir_path, "epoch_result.json"), "w"
        ) as f:
            json.dump(epoch_result, f, indent=4)
        return epoch_result


    def evaluate(self, data_loader, metrics, *args, **kwargs):

        return 0


