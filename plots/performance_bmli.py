import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import argparse
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
args = parser.parse_args()

if args.study == 1:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_ranking.pth")
    _, performance_baselines, _ = torch.load("data/baselines_ranking.pth")
    _, performance_rr, _ = torch.load('data/bmli_ranking.pth')
    print(performance_rr.shape)
    #print(avg_performance.mean(1).mean(1))
elif args.study == 2:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_direction.pth")
    _, performance_baselines, _ = torch.load("data/baselines_direction.pth")
    _, performance_rr, _ = torch.load('data/bmli_direction.pth')
elif args.study == 3:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_none.pth")
    _, performance_baselines, _ = torch.load("data/baselines_none.pth")
    _, performance_rr, _ = torch.load('data/bmli_none.pth')

performance_baselines = performance_baselines.detach()
performance_baselines = performance_baselines[[0, 2, 1]]
performance_baselines = performance_baselines.numpy().mean(2) # average over episodes
means_baselines = performance_baselines.mean(axis=1) # average over runs
correct_reponses = (targets == predictions).view(targets.shape[0],targets.shape[1], -1).float()

means_rr = performance_rr.mean(1).mean(1)[[0, 2, 4, 6]]
mpl.rcParams['figure.figsize'] = (6.5, 4)

styles = ['-',  (0, (5, 7)), (0, (5, 5)), (0, (5, 3)), (0, (5, 1))]
plt.plot(np.arange(0, 10), means_baselines[0], c='black', alpha=0.7, ls=styles[0])
for i in range(means_rr.shape[0]):
    plt.plot(np.arange(0, 10), means_rr[i], c='black', alpha=0.7, ls=styles[i+1])

plt.legend(['Ideal Observer', 'BMLI ' + r'($\beta = 0.1$)','BMLI ' + r'($\beta = 0.01$)', 'BMLI ' + r'($\beta = 0.001$)', 'MLI'], frameon=False, loc='upper left') #
plt.ylim(0.5, 1)
plt.xlim(0, 9)
plt.xlabel('Trial')
plt.ylabel('Accuracy')
sns.despine()
plt.tight_layout()
plt.savefig('plots/figures/performance_bmli' + str(args.study) + '.pdf', bbox_inches='tight')
plt.show()
