import torch
import torch.nn as nn

from einops import rearrange

import torch.nn.functional as F
import numpy as np

from models.OneNet_ODG.layer_time import DilatedConvEncoder_time
from models.OneNet_ODG.layer import DilatedConvEncoder



class TempoEnc(nn.Module):
    def __init__(
        self,
        n_time,
        n_attr,
        normal=True
    ):
        super().__init__()

        self.time = n_time
        self.enc = nn.Embedding(n_time, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)

    def forward(self, x, start=0, t_left=None):
        length = x.shape[-2]
        if t_left == None:
            enc = self.enc(torch.arange(start, start + length).cuda())
        else:
            enc = self.enc(torch.Tensor(t_left).long().cuda())
        x = x + enc
        if self.no:
            x = self.norm(x)
        return x


class timestamp(nn.Module):
    def __init__(
        self, history_len, emb_channels
    ):
        super().__init__()
        self.time_stamp = nn.Embedding(24, emb_channels)
        # add temporal embedding and normalize
        self.tempral_enc = TempoEnc(history_len, emb_channels, True)

    def forward(self, stamp):
        time_emb = self.time_stamp(stamp)
        time_emb = self.tempral_enc(time_emb)
        return time_emb


def laplacian(W):
    N, N = W.shape
    W = W+torch.eye(N).to(W.device)
    D = W.sum(axis=1)
    D = torch.diag(D**(-0.5))
    out = D@W@D
    return out


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder_time(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder_time(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


class TSEncoder(nn.Module):
    def __init__(self, device, num_nodes, supports, history_len, in_dim=2, out_dim=12, residual_channels = 32, dilation_channels = 32, skip_channels=256, end_channels=512, emb_channels = 16, blocks=4, layers=2, dropout=0.3, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.mask_mode = mask_mode
        self.supports = supports
        self.supports_len = len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1
        self.adp = torch.zeros(num_nodes, num_nodes).to(device)
        kt = 3
        self.feature_extractor = DilatedConvEncoder(
            residual_channels,
            dilation_channels,
            skip_channels,
            blocks,
            layers,
            dropout,
            kt,
            self.supports_len,
            kernel_size=2,
            gamma=gamma
        )

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.reduce_stamp = nn.Linear(history_len, 1, bias=False)  # 平均
        self.temp_1 = nn.Linear(emb_channels, kt + 1)
        self.kt = kt


    def forward(self, x, time_emb, mask):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        supports = self.supports + [adp]
        #生成矩阵多项式系数
        x = x.transpose(1, 2)
        period_emb = self.reduce_stamp(time_emb.permute(0, 2, 1)).squeeze(2)
        temp_1 = self.temp_1(period_emb)
        x = x.transpose(1, 2)
        x = x.transpose(1, 3)
        x = nn.functional.pad(x, (1, 0, 0, 0))

        x = self.start_conv(x)
        x = self.feature_extractor(x, supports, temp_1)
        x = F.relu(self.end_conv_1(x))

        x = self.end_conv_2(x)

        return x





class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input, time_emb):
        return self.encoder(input, time_emb, mask=self.mask)

class TS2VecEncoderWrapper_time(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)


class onenet_odg(nn.Module):
    def __init__(self, num_nodes, adj_mx, history_len, in_dim, out_dim, residual_channels, dilation_channels, skip_channels, end_channels, emb_channels, blocks, layers, dropout, gamma, device):
        super().__init__()
        self.device = device
        if adj_mx != None:
            supports = [torch.tensor(i).to(device) for i in adj_mx]
        else:
            supports = []

        depth = 10
        encoder = TSEncoder_time(input_dims=history_len,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=depth)
        self.encoder_time = TS2VecEncoderWrapper_time(encoder, mask='all_true').to(self.device)
        self.regressor_time = nn.Linear(320, out_dim).to(self.device)

        encoder = TSEncoder(device, num_nodes, supports, history_len,in_dim =in_dim, out_dim=out_dim, residual_channels = residual_channels, dilation_channels = dilation_channels, skip_channels=skip_channels, end_channels=end_channels, emb_channels= emb_channels, blocks=blocks, layers=layers, dropout=dropout, gamma=gamma)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.stamp_emb = timestamp(history_len, emb_channels)
        self.num_nodes = num_nodes
        self.emb_channels = emb_channels

    def forward_individual(self, x, stamp):
        rep = self.encoder_time.encoder.forward_time(x)
        y = self.regressor_time(rep).transpose(1, 2)
        y1 = rearrange(y, 'b t d -> b (t d)')

        time_emb = self.stamp_emb(stamp)
        y2 = self.encoder(x, time_emb)

        return y1, y2

    def forward_weight(self, x, stamp, g1, g2):
        rep = self.encoder_time.encoder.forward_time(x[:, :, :, 0])
        y = self.regressor_time(rep).transpose(1, 2)
        y1 = rearrange(y, 'b t d -> b (t d)')

        time_emb = self.stamp_emb(stamp)
        y2 = self.encoder(x, time_emb)
        y2 = rearrange(y2, 'b t n d -> b (t n d)')
        return y1.detach() * g1 + y2.detach() * g2, y1, y2


    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
