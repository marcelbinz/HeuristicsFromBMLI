from models.baselines import VariationalProbitRegression, VariationalFirstCue, VariationalFirstDiscriminatingCue
from models.environments import PairedComparison
import torch
from torch.distributions import kl_divergence
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute or not')
args = parser.parse_args()

if args.recompute:
    num_steps = 10
    num_tasks = 10000

    kl = torch.zeros(2, num_tasks, num_steps)

    for k, dichotomized in enumerate([True, False]):
        data_loader = PairedComparison(4, direction=False, dichotomized=dichotomized, ranking=True)
        for i in tqdm(range(num_tasks)):
            if dichotomized:
                model1 = VariationalFirstDiscriminatingCue(data_loader.num_inputs)
            else:
                model1 = VariationalFirstCue(data_loader.num_inputs)
            model2 = VariationalProbitRegression(data_loader.num_inputs)
            inputs, targets, _, _ = data_loader.get_batch(1, num_steps)

            predictive_distribution1 = model1.forward(inputs, targets)
            predictive_distribution2 = model2.forward(inputs, targets)
            kl[k, i, :] =  kl_divergence(predictive_distribution1, predictive_distribution2).squeeze()

    torch.save(kl, 'data/power_analysis.pth')
    print(kl.mean(1))
    print(kl.mean(1).sum(-1))
else:
    kl = torch.load('data/power_analysis.pth').div(2.303) # ln to log10
    mean = kl.sum(-1).mean(1)
    variance = kl.sum(-1).std(1).pow(2)

    num_tasks = torch.arange(1, 31)
    expected_bf = torch.arange(1, 31).unsqueeze(-1) * mean
    confidence_bf = (torch.arange(1, 31).unsqueeze(-1) * variance).sqrt() * 1.96
    styles = [':', '-']
    for i in range(expected_bf.shape[1]):
        plt.plot(num_tasks, expected_bf[:, i].detach(), color='black', ls=styles[i])
    plt.xlabel('Number of Tasks', fontsize=12)
    plt.ylabel('Expected Log Bayes Factor', fontsize=12)
    plt.xlim(1, 30)
    plt.ylim(0, 3.2)
    sns.despine()
    plt.tight_layout()
    plt.legend(['Dichotomized', 'Continuous'], frameon=False)
    plt.savefig('plots/figures/power_analysis.pdf', bbox_inches='tight')
    plt.show()
