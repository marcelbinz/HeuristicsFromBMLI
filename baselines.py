import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.environments import PairedComparison
from tqdm import tqdm
import random

def gini(x):
    # https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

class Guessing():
    def __init__(self, num_features, noise=0):
        self.num_features = num_features

    def forward(self, inputs, targets):
        return Bernoulli(0.5 * torch.ones_like(targets))

class GRUNoHidden(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(GRUNoHidden, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden

        self.weight_ih = nn.Linear(num_inputs, 3 * num_hidden)
        self.linear_mu = nn.Linear(num_hidden, num_outputs)
        self.linear_logscale = nn.Linear(num_hidden, num_outputs)


    def forward(self, inputs):
        gi = self.weight_ih(inputs)
        i_r, i_i, i_n = gi.chunk(3, 1)

        resetgate = torch.sigmoid(i_r)
        inputgate = torch.sigmoid(i_i)
        newgate = torch.tanh(i_n + resetgate)
        hy = newgate - inputgate * (newgate)
        mean = self.linear_mu(hy)
        log_sigma = self.linear_logscale(hy)

        return mean.t(), log_sigma.t()


class FeedForward(nn.Module):
    def __init__(self, num_features, noise=1.0, learning_rate=0.1):
        super(FeedForward, self).__init__()

        self.network = GRUNoHidden(num_features, num_features, 128)
        self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)

        self.noise = noise * torch.ones(1)

    def forward(self, inputs, targets):
        sequence_length = inputs.shape[0]
        thetas = []
        self.means = []
        for t in range(sequence_length):
            with torch.no_grad():
                theta, mean = self.predictive_distribution(inputs[t, :])
                thetas.append(theta.clone())
                self.means.append(mean.clone())

            self.optimizer.zero_grad()
            loss = self.loss(inputs[t, :], targets[t, :])
            loss.backward()
            self.optimizer.step()

        self.means = torch.stack(self.means)

        return Bernoulli(torch.stack(thetas))

    def loss(self, X, y, n_samples=100):
        mean, log_sigma = self.network(X)
        q = Normal(mean, torch.exp(log_sigma))

        weights = q.rsample((n_samples,))
        theta =  Normal(0, 1).cdf(X @ weights / (math.sqrt(2) * self.noise))
        predictive_distribution = Bernoulli(theta)

        return -predictive_distribution.log_prob(y).mean()

    def predictive_distribution(self, X):
        mean, log_sigma = self.network(X)
        theta = Normal(0, 1).cdf(X @ mean / (math.sqrt(2 * self.noise.pow(2) + (X.pow(2) @ torch.exp(log_sigma).pow(2)))))
        return theta, mean

class VariationalProbitRegression(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super(VariationalProbitRegression, self).__init__()
        self.max_iters = max_iters

        self.mean = nn.Parameter(torch.zeros(num_features, 1))
        self.log_sigma = nn.Parameter(torch.zeros(num_features, 1))

        self.prior_mean = torch.zeros(num_features, 1)
        self.prior_log_sigma = torch.zeros(num_features, 1)

        self.noise = noise * torch.ones(1)

    def forward(self, inputs, targets):
        sequence_length = inputs.shape[0]
        thetas = []
        self.weights = []
        self.losses = []
        for t in range(sequence_length):
            self.weights.append(self.mean.clone())
            theta, weights = self.predictive_distribution(inputs[t, :])
            thetas.append(theta.clone())
            optimizer = optim.Adam(self.parameters(), lr=0.1, amsgrad=True)
            lowest_loss = math.inf
            for step in range(self.max_iters):
                optimizer.zero_grad()
                loss = self.loss(inputs[t, :], targets[t, :])
                if step > 0 and loss.item() >= lowest_loss:
                    counter += 1
                else:
                    if loss.item() < lowest_loss:
                        lowest_loss = loss.item()
                    counter = 1
                if counter == 10:
                    break
                loss.backward()
                old_loss = loss
                optimizer.step()

            self.losses.append(loss.clone())
            self.update_prior()
        self.weights = torch.stack(self.weights)
        return Bernoulli(torch.stack(thetas))

    def early_stopping(self, threshold):
        total_norm = torch.norm(torch.cat([x.grad.flatten() for x in self.parameters()])).detach()
        return total_norm < threshold

    def loss(self, X, y, n_samples=100):
        p = Normal(self.prior_mean, torch.exp(self.prior_log_sigma))
        q = Normal(self.mean, torch.exp(self.log_sigma))

        weights = q.rsample((n_samples,))
        theta =  Normal(0, 1).cdf(X @ weights / (math.sqrt(2) * self.noise))

        predictive_distribution = Bernoulli(theta)

        nll = -predictive_distribution.log_prob(y).mean()
        kld =  kl_divergence(q, p).sum()
        loss = nll.mean() + kld

        return loss

    def predictive_distribution(self, X):
        theta = Normal(0, 1).cdf(X @ self.mean / (math.sqrt(2 * self.noise.pow(2) + (X.pow(2) @ torch.exp(self.log_sigma).pow(2)))))
        return theta, self.mean

    def update_prior(self):
        self.prior_mean = self.mean.data.clone()
        self.prior_log_sigma = self.log_sigma.data.clone()


class VariationalEqualWeighting(VariationalProbitRegression):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super().__init__(num_features=1, max_iters=max_iters, noise=noise)

    def forward(self, inputs, targets):
        return super().forward(inputs.sum(-1, keepdim=True), targets)

class VariationalFirstCue(VariationalProbitRegression):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super().__init__(num_features=1, max_iters=max_iters, noise=noise)

    def forward(self, inputs, targets):
        return super().forward(inputs[:, :, [0]], targets)

class VariationalThresholdFirstCue(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0, delta=0.2):
        super(VariationalThresholdFirstCue, self).__init__()
        self.models = nn.ModuleList([VariationalProbitRegression(num_features=1, max_iters=max_iters, noise=noise) for i in range(num_features)])
        self.losses = None
        self.max_iters = max_iters
        self.delta = delta

    def forward(self, inputs, targets):
        thetas = []
        accumulated_evidence = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs[:, :, [i]], targets).probs)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        for t in range(1, theta.shape[0]):
            for k in range(inputs.shape[-1]):
                if torch.abs(inputs[t, 0, k]) > self.delta:
                    theta[t] = thetas[k][t]
                    break

        return Bernoulli(theta)

class VariationalThresholdBestCue(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0, delta=0.2):
        super(VariationalThresholdBestCue, self).__init__()
        self.models = nn.ModuleList([VariationalProbitRegression(num_features=1, max_iters=max_iters, noise=noise) for i in range(num_features)])
        self.losses = None
        self.max_iters = max_iters
        self.delta = delta

    def forward(self, inputs, targets):
        thetas = []
        accumulated_evidence = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs[:, :, [i]], targets).probs)
            accumulated_evidence.append(torch.cumsum(Bernoulli(thetas[i]).log_prob(targets).squeeze(), dim=-1))
        accumulated_evidence = torch.stack(accumulated_evidence)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        for t in range(1, theta.shape[0]):
            if t == 1:
                best_model_index = random.choice(range(len(self.models)))
                theta[t] = sum(thetas)[t] / len(thetas)
            else:
                for k in torch.argsort(accumulated_evidence[:, t-1], descending=True):
                    if torch.abs(inputs[t, 0, k.item()]) > self.delta:
                        theta[t] = thetas[k.item()][t]
                        break

        return Bernoulli(theta)

class VariationalBestCue(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super(VariationalBestCue, self).__init__()
        self.models = nn.ModuleList([VariationalProbitRegression(num_features=1, max_iters=max_iters, noise=noise) for i in range(num_features)])
        self.losses = None
        self.max_iters = max_iters

    def forward(self, inputs, targets):
        thetas = []
        accumulated_evidence = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs[:, :, [i]], targets).probs)
            accumulated_evidence.append(torch.cumsum(Bernoulli(thetas[i]).log_prob(targets).squeeze(), dim=-1))
        accumulated_evidence = torch.stack(accumulated_evidence)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        for t in range(1, theta.shape[0]):
            if t == 1:
                best_model_index = random.choice(range(len(self.models)))
                theta[t] = sum(thetas)[t] / len(thetas)
            else:
                best_model_index = torch.argmax(accumulated_evidence[:, t-1])
                theta[t] = thetas[best_model_index][t]

        return Bernoulli(theta)

class StrategySelection(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0, ranking=True):
        super(StrategySelection, self).__init__()
        if ranking:
            self.models = nn.ModuleList([VariationalProbitRegression(num_features=num_features, max_iters=max_iters, noise=noise),
                VariationalEqualWeighting(num_features=num_features, max_iters=max_iters, noise=noise),
                VariationalFirstCue(num_features=num_features, max_iters=max_iters, noise=noise)
            ])
        else:
            self.models = nn.ModuleList([VariationalProbitRegression(num_features=num_features, max_iters=max_iters, noise=noise),
                VariationalEqualWeighting(num_features=num_features, max_iters=max_iters, noise=noise),
                VariationalBestCue(num_features=num_features, max_iters=max_iters, noise=noise)
            ])
        self.losses = None
        self.max_iters = max_iters

    def forward(self, inputs, targets):
        thetas = []
        accumulated_evidence = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs, targets).probs)
            accumulated_evidence.append(torch.cumsum(Bernoulli(thetas[i]).log_prob(targets).squeeze(), dim=-1))
        accumulated_evidence = torch.stack(accumulated_evidence)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        self.selected_model = torch.zeros(targets.shape[0], len(self.models))
        self.selected_model[:2, :] = 1/3
        for t in range(1, theta.shape[0]):
            if t == 1:
                best_model_index = random.choice(range(len(self.models)))
                theta[t] = sum(thetas)[t] / len(thetas)
            else:
                best_model_index = torch.argmax(accumulated_evidence[:, t-1])
                self.selected_model[t, best_model_index] += 1
                theta[t] = thetas[best_model_index][t]

        return Bernoulli(theta)

class VariationalFirstDiscriminatingCue(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super(VariationalFirstDiscriminatingCue, self).__init__()
        self.models = nn.ModuleList([VariationalProbitRegression(num_features=1, max_iters=max_iters, noise=noise) for i in range(num_features)])
        self.max_iters = max_iters

    def forward(self, inputs, targets):
        thetas = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs[:, :, [i]], targets).probs)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        for t in range(theta.shape[0]):
            for j in range(len(self.models)):
                if inputs[t, 0, j] != 0:
                    theta[t] = thetas[j][t]
                    break

        return Bernoulli(theta)
