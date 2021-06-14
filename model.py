import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import math

# https://github.com/senya-ashukha/sparse-vd-pytorch/blob/master/svdo-solution.ipynb
class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def sample_weights(self):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            self.sampled_weights = Normal(self.W, torch.exp(self.log_sigma) + 1e-8).rsample()
        else:
            self.sampled_weights = self.W * (self.log_alpha < 3).float()

    def reset_log_sigma(self):
        self.log_sigma.data.fill_(-5)

    def reset_parameters(self):
        nn.init.orthogonal_(self.W)
        self.bias.data.fill_(0)
        self.reset_log_sigma()

    def forward(self, x):
        return F.linear(x, self.sampled_weights, self.bias)

    def kl_reg(self):
        # Return KL here -- a scalar
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha)) - k1
        a = - torch.sum(kl)
        return a

class InitialState(nn.Module):
    def __init__(self, features):
        super(InitialState, self).__init__()
        self.state = nn.Parameter(torch.zeros(1, features), requires_grad=True)

    def forward(self, batch_size):
        return self.state.expand(batch_size, -1)

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = LinearSVDO(input_size, 3 * hidden_size)
        self.weight_hh = LinearSVDO(hidden_size, 3 * hidden_size)

        for i in range(3):
            nn.init.orthogonal_(self.weight_hh.W.data[(i*self.hidden_size):((i+1)*self.hidden_size), :])
            nn.init.orthogonal_(self.weight_ih.W.data[(i*self.hidden_size):((i+1)*self.hidden_size), :])

    def forward(self, inputs, hidden):
        gi = self.weight_ih(inputs)
        gh = self.weight_hh(hidden)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy

class GRU(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(GRU, self).__init__()

        assert num_outputs == 1

        self.cell = GRUCell(num_inputs + 2, num_hidden)
        self.initial_states = InitialState(num_hidden)
        self.linear_mu = LinearSVDO(num_hidden, num_inputs)
        self.linear_logscale = LinearSVDO(num_hidden, num_inputs)
        self.noise = nn.Parameter(0.1 * torch.ones(1), requires_grad=False)

    def forward(self, si, st, sampling=False):
        time_steps = (torch.arange(si.shape[0]).float())[:, None, None].expand(-1, si.shape[1], -1).to(si.device)
        inputs = torch.cat((si, st, 1 - st), -1)

        time_steps = inputs.shape[0] - 1
        batch_size = inputs.shape[1]

        outputs = []
        hidden = self.initial_states(batch_size)
        outputs.append(hidden.clone())

        self.cell.weight_ih.sample_weights()
        self.cell.weight_hh.sample_weights()
        self.linear_mu.sample_weights()
        self.linear_logscale.sample_weights()

        for i in range(time_steps):
            hidden = self.cell(inputs[i], hidden)
            outputs.append(hidden.clone())
        outputs = torch.stack(outputs, dim=0)

        m = self.linear_mu(outputs)
        s = torch.exp(self.linear_logscale(outputs))
        if sampling:
            q = Normal(m, s)
            w = q.rsample()
            theta = Normal(0, 1).cdf((si * w).sum(-1, keepdim=True) / (math.sqrt(2) * self.noise))
        else:
            theta = Normal(0, 1).cdf((si * m).sum(-1, keepdim=True) / (torch.sqrt(2 * self.noise.pow(2) + (si.pow(2) * s.pow(2)).sum(-1, keepdim=True))))

        return Bernoulli(theta), m, s

    def reset_log_sigma(self):
        self.linear_mu.reset_log_sigma()
        self.linear_logscale.reset_log_sigma()
        self.cell.weight_ih.reset_log_sigma()
        self.cell.weight_hh.reset_log_sigma()

    def regularization(self, alpha):
        penalty = self.linear_mu.kl_reg() + self.linear_logscale.kl_reg() + self.cell.weight_ih.kl_reg() + self.cell.weight_hh.kl_reg()

        return alpha * penalty
