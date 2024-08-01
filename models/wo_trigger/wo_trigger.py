import torch
import torch.nn as nn


import torch.nn.functional as F

from models.wo_trigger.layer import DilatedConvEncoder

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
        self, history_len, emb_channels, steps_per_day
    ):
        super().__init__()
        self.time_stamp = nn.Embedding(steps_per_day, emb_channels)
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


class TSEncoder(nn.Module):
    def __init__(self, device, num_nodes, supports, history_len, in_dim=2, out_dim=12, residual_channels = 32, dilation_channels = 32, skip_channels=256, end_channels=512, emb_channels = 16, blocks=4, layers=2, dropout=0.3, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.mask_mode = mask_mode
        self.supports = supports
        self.supports_len = len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1

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


class wo_trigger(nn.Module):
    def __init__(self, num_nodes, adj_mx, history_len, in_dim, out_dim, residual_channels, dilation_channels, skip_channels, end_channels, emb_channels, blocks, layers, dropout, gamma, device, steps_per_day):
        super().__init__()
        self.device = device
        if adj_mx != None:
            supports = [torch.tensor(i).to(device) for i in adj_mx]
        else:
            supports = []
        encoder = TSEncoder(device, num_nodes, supports, history_len,in_dim =in_dim, out_dim=out_dim, residual_channels = residual_channels, dilation_channels = dilation_channels, skip_channels=skip_channels, end_channels=end_channels, emb_channels= emb_channels, blocks=blocks, layers=layers, dropout=dropout, gamma=gamma)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.stamp_emb = timestamp(history_len, emb_channels, steps_per_day)
        self.line_stamp = nn.Linear(num_nodes, emb_channels)
    def forward(self, x, stamp):
        time_emb = self.stamp_emb(stamp)
        return self.encoder(x, time_emb)

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()

