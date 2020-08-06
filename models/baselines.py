import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.environments import PairedComparison
from tqdm import tqdm

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
    def __init__(self, num_features):
        self.num_features = num_features

    def forward(self, inputs, targets):
        return Bernoulli(0.5 * torch.ones_like(targets))

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

class VariationalBestCue(nn.Module):
    def __init__(self, num_features, max_iters=1000, noise=1.0):
        super(VariationalBestCue, self).__init__()
        self.models = nn.ModuleList([VariationalProbitRegression(num_features=1, max_iters=max_iters, noise=noise) for i in range(num_features)])
        self.losses = None
        self.max_iters = max_iters

    def forward(self, inputs, targets):
        thetas = []
        for i in range(len(self.models)):
            thetas.append(self.models[i].forward(inputs[:, :, [i]], targets).probs)

        accumulated_elbo = -1.0 * torch.cumsum(torch.Tensor([m.losses for m in self.models]), dim=-1)

        theta = 0.5 * torch.ones_like(targets) # seq_len, 1, 1
        for t in range(1, theta.shape[0]):
            best_model_index = torch.argmax(accumulated_elbo[:, t-1])
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

if __name__ == '__main__':

    num_experiments = 100
    num_episodes = 30
    sequence_length = 10

    for mode in [0, 1, 2]:
        if mode == 0:
            save_path = 'data/baselines_ranking.pth'
            models = [VariationalProbitRegression, VariationalEqualWeighting, VariationalFirstCue]
            ranking = True
            direction = False
        elif mode == 1:
            save_path = 'data/baselines_direction.pth'
            models = [VariationalProbitRegression, VariationalEqualWeighting, VariationalBestCue]
            ranking = False
            direction = True
        elif mode == 2:
            save_path = 'data/baselines_none.pth'
            models = [VariationalProbitRegression, VariationalEqualWeighting, VariationalBestCue]
            ranking = False
            direction = False

        data_loader = PairedComparison(4, ranking=ranking, direction=direction, dichotomized=False)

        map_performance = torch.zeros(len(models), num_experiments, num_episodes, sequence_length)
        avg_performance = torch.zeros(len(models), num_experiments, num_episodes, sequence_length)
        gini_coefficients = torch.zeros(1, num_experiments, num_episodes, sequence_length)

        for j, model_class in enumerate(models):
            for k in tqdm(range(num_experiments)):
                for i in range(num_episodes):
                    inputs, targets, _, _ = data_loader.get_batch(1, sequence_length)
                    model = model_class(data_loader.num_inputs, noise=data_loader.sigma)
                    predictive_distribution = model.forward(inputs, targets)

                    prediction = (predictive_distribution.probs > 0.5).float()
                    map_performance[j, k, i] = (prediction == targets).squeeze()
                    avg_performance[j, k, i] = ((1 - targets) * (1 - predictive_distribution.probs) + targets * predictive_distribution.probs).squeeze()

                    if j == 0:
                        for t in range(sequence_length):
                            gini_coefficients[j, k, i, t] = gini(torch.abs(model.weights[t].t()).squeeze().detach().cpu().numpy())

        print(map_performance.mean(1).mean(1))
        print(avg_performance.mean(1).mean(1))
        torch.save([map_performance, avg_performance, gini_coefficients], save_path)
