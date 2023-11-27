import torch
import torch.nn as nn
import math


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.hidden_size = hidden_size

        self.weight_ir = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ir, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hr, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ir)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_r, -bound, bound)

    def forward(self, x, hidden):
        g = torch.sigmoid(x @ self.weight_ir.t() + hidden @ self.weight_hr.t() + self.bias_r)
        o = torch.cat((g * x, (1-g) * hidden), dim=-1)
        return o

