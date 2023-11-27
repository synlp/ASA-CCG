# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** 0.5

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.w_k.weight)
        nn.init.orthogonal_(self.w_v.weight)

    def forward(self, x):
        k = self.w_k(x)
        v = self.w_v(x)
        weight = nn.functional.softmax(torch.bmm(k, v.transpose(1, 2)) / self.scale, dim=2)
        y = torch.bmm(weight, x)
        return y
