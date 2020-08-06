import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
parser.add_argument('--bw', action='store_true', default=False, help='black and white plotting')
args = parser.parse_args()

if args.study == 1:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_ranking.pth")
    _, performance_baselines, _ = torch.load("data/baselines_ranking.pth")
    _, performance_rr, _ = torch.load('data/bmli_ranking.pth')
elif args.study == 2:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_direction.pth")
    _, performance_baselines, _ = torch.load("data/baselines_direction.pth")
    _, performance_rr, _ = torch.load('data/bmli_direction.pth')
elif args.study == 3:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_none.pth")
    _, performance_baselines, _ = torch.load("data/baselines_none.pth")
    _, performance_rr, _ = torch.load('data/bmli_none.pth')

performance_baselines = performance_baselines.detach()
# reorder baselines
performance_baselines = performance_baselines[[0, 2, 1]]
# average over episodes
performance_baselines = performance_baselines.numpy().mean(2)
 # average over runs
means_baselines = performance_baselines.mean(axis=1)
correct_reponses = (targets == predictions).view(targets.shape[0],targets.shape[1], -1).float()

rr_index = 2
means_rr = performance_rr.mean(1).mean(1)[rr_index]

print("Mean accuracy for all participant:")
participant_mean = correct_reponses.mean(-1).mean(-1)
print(participant_mean.mean())
print(participant_mean.std())

print("Mean accuracy over time:")
mean_performance = correct_reponses.mean(-1).mean(0)
std_performance = correct_reponses.mean(-1).std(0)
print(mean_performance)

# matplotlib settings
mpl.rcParams['figure.figsize'] = (6.5, 4)
if args.bw:
    colors = ['black', 'black', 'black', 'black', 'black']
    styles = ['-','-','-','-','-']
    markers = ['o','^','s','*',',']
else:
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd', '#2ca02c']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    markers = [',',',',',',',',',']

# plot
for i in range(means_baselines.shape[0]):
    plt.plot(np.arange(0, 10), means_baselines[i], c=colors[i], alpha=0.7, ls=styles[i], marker=markers[i])
plt.plot(np.arange(0, 10), means_rr, c=colors[3], alpha=0.7, ls=styles[3], marker=markers[3])
plt.plot(np.arange(0, 10), mean_performance, c=colors[4], alpha=0.7, ls=styles[4], marker=markers[4])
plt.fill_between(np.arange(0, 10), mean_performance-(std_performance / math.sqrt(correct_reponses.shape[0])), mean_performance+(std_performance  / math.sqrt(correct_reponses.shape[0])), color=colors[4], alpha=0.1)

plt.legend(['Ideal Observer', 'Single Cue', 'Equal Weighting', 'BMLI ' + r'($\beta = 0.01$)', 'Humans'], frameon=False)
plt.ylim(0.5, 1)
plt.xlim(0, 9)
plt.xlabel('Trial')
plt.ylabel('Accuracy')
sns.despine()
plt.tight_layout()
plt.savefig('plots/figures/performance' + str(args.study) + '.pdf', bbox_inches='tight')
plt.show()
