import math
import torch
from pyro.distributions.lkj import LKJCorrCholesky
from torch.distributions import MultivariateNormal

class PairedComparison():
    def __init__(self, num_inputs=4, num_targets=1, direction=False, ranking=False, dichotomized=False):
        self.num_inputs = num_inputs
        self.num_targets = num_targets

        self.direction = direction
        self.ranking = ranking
        self.dichotomized = dichotomized

        self.sigma = math.sqrt(0.01)
        self.theta = 1.0 * torch.ones(num_inputs)
        self.cov_prior = LKJCorrCholesky(num_inputs, eta=2.0 * torch.ones(1))

    def sample_pair(self, weights, L):
        L = L.squeeze()
        inputs_a = MultivariateNormal(torch.zeros(self.num_inputs), scale_tril=torch.mm(torch.diag(torch.sqrt(self.theta)), L)).sample()
        inputs_b = MultivariateNormal(torch.zeros(self.num_inputs), scale_tril=torch.mm(torch.diag(torch.sqrt(self.theta)), L)).sample()
        if self.dichotomized:
            inputs_a = (inputs_a > 0).float()
            inputs_b = (inputs_b > 0).float()
        inputs = inputs_a - inputs_b
        targets = torch.bernoulli(0.5 * torch.erfc(-(weights * inputs).sum(-1, keepdim=True) / (2 * self.sigma)))

        return inputs, targets, inputs_a, inputs_b

    def get_batch(self, batch_size, num_supports, device=None):
        support_inputs = torch.zeros(num_supports, batch_size, self.num_inputs)
        support_inputs_a = torch.zeros(num_supports, batch_size, self.num_inputs)
        support_inputs_b = torch.zeros(num_supports, batch_size, self.num_inputs)
        support_targets = torch.zeros(num_supports, batch_size, self.num_targets)
        self.weights = torch.zeros(batch_size, self.num_inputs)

        for i in range(batch_size):
            if self.direction:
                weights = torch.randn(self.num_inputs).abs()
            else:
                weights = torch.randn(self.num_inputs)

            if self.ranking:
                absolutes = torch.abs(weights)
                _, feature_perm = torch.sort(absolutes, dim=0, descending=True)
                weights = weights[feature_perm]

            L = self.cov_prior.sample()
            self.weights[i] = weights.clone()
            for j in range(num_supports):
                support_inputs[j, i], support_targets[j, i], support_inputs_a[j, i], support_inputs_b[j, i] = self.sample_pair(weights, L)

        # this is a shitty hack to fix an earlier bug
        if self.direction:
            support_targets = 1 - support_targets

        return support_inputs.detach().to(device), support_targets.detach().to(device), support_inputs_a.detach().to(device), support_inputs_b.detach().to(device)

if __name__ == '__main__':
    dl = PairedComparison(4, dichotomized=False, direction=False, ranking=False)
    for i in range(1):
        inputs, targets, _, _ = dl.get_batch(1000, 10)
        print(targets.mean())
        print(inputs.std() ** 2)
