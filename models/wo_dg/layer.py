import gc

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

from itertools import chain

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W/ W_norm
    return W


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, supports_len, kernel_size, dropout, dilation=1, groups=1, gamma=0.9):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        #padding = self.receptive_field // 2 暂时去掉


        self.conv = nn.Conv2d(
            in_channels, out_channels, (1,kernel_size),
            #padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.gconv = gcn(out_channels, in_channels, dropout, support_len=supports_len)
        self.skip_conv = nn.Conv1d(in_channels=in_channels, out_channels=skip_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(in_channels)


        self.bias = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad=True)
        #self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)

        self.in_channels = in_channels
        self.out_features = out_channels

        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels * kernel_size // self.n_chunks)

        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        nh = 64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels // in_channels)
        self.calib_f = nn.Linear(nh, out_channels // in_channels)
        dim = self.n_chunks * (self.chunk_out_d + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)

        # self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        # self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        # self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75

    def store_grad(self):
        # print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1 - self.f_gamma) * grad
        if not self.training:
            e = self.cos(self.f_grads, self.grads)
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1 - self.gamma) * grad

    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)])
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().cuda())
        else:
            self.q_ema = self.f_gamma * self.q_ema + (1 - self.f_gamma) * q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0)
            self.trigger = 0
            # read

            att = q @ self.W
            att = F.softmax(att / 0.5)
            v, idx = torch.topk(att, 2)
            ww = torch.index_select(self.W, 1, idx)
            idx = idx.unsqueeze(1).float()
            old_w = ww @ idx
            # write memory
            s_att = torch.zeros(att.size(0)).cuda()
            s_att[idx.squeeze().long()] = v.squeeze()
            W = old_w @ s_att.unsqueeze(0)
            mask = torch.ones(W.size()).cuda()
            mask[:, idx.squeeze().long()] = self.tau
            self.W.data = mask * self.W.data + (1 - mask) * W
            self.W.data = normalize(self.W.data)
            # retrieve
            ll = torch.split(old_w, dim)
            nw, nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw + nb]]), torch.cat(*[ll[-nf:]])

            try:
                w = self.tau * w + (1 - self.tau) * o_w.view(w.size())
                b = self.tau * b + (1 - self.tau) * o_b.view(b.size())
                f = self.tau * f + (1 - self.tau) * o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        return w.unsqueeze(0).unsqueeze(-2), b.view(-1), f

    def forward(self, x, skip, supports):
        w, b, f = self.fw_chunks()
        cw = self.conv.weight * w
        residual = x
        conv_out = F.conv2d(x, cw, dilation=self.dilation, bias=self.bias * b)
        x = f * conv_out
        s = x
        s = self.skip_conv(s)
        try:
            skip = skip[:, :, :, -s.size(3):]
        except:
            skip = 0
        skip = s + skip

        x = self.gconv(x, supports)
        x = x + residual[:, :, :, -x.size(3):]
        out = self.bn(x)

        return out, skip




    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, layers, supports_len, kernel_size, dilation, dropout=0.3, gamma=0.9):
        super().__init__(),

        self.convs = nn.ModuleList()
        self.layers = layers

        for i in range(self.layers):
            self.convs.append(SamePadConv(in_channels, out_channels, skip_channels, supports_len, kernel_size, dilation=dilation, dropout = dropout, gamma=gamma))
            dilation *= 2
    def forward(self, x, skip, supports):
        for i in range(self.layers):
            x = F.gelu(x)
            x, skip = self.convs[i](x, skip, supports)
        return x, skip

class DilatedConvEncoder(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, blocks, layers, dropout, supports_len, kernel_size, gamma):
        super().__init__()

        self.blocks = blocks
        self.nets = nn.ModuleList()
        for i in range(self.blocks):
            self.nets.append(ConvBlock(
                residual_channels,
                dilation_channels,
                skip_channels,
                layers,
                supports_len,
                kernel_size=kernel_size,
                dilation=1,
                dropout = dropout,
                gamma=gamma
            ))


    def forward(self, x, supports):
        skip = 0
        for i in range(self.blocks):
            x, skip = self.nets[i](x, skip, supports)
        x = F.relu(skip)
        return x
        

