import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
parser.add_argument('--bw', action='store_true', default=False, help='black and white plotting')
args = parser.parse_args()

mpl.rcParams['figure.figsize'] = (3.0, 2.5)

if args.study == 1:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_ranking.pth")
elif args.study == 2:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_direction.pth")
elif args.study == 3:
    inputs_a, inputs_b, targets, predictions, _ = torch.load("data/humans_none.pth")

threshold = 0.0

correct_reponses = (targets == predictions).view(targets.shape[0],targets.shape[1], -1).float()
correct_reponses1 = correct_reponses[:, :, :15]
correct_reponses2 = correct_reponses[:, :, 15:]

mean_performance1 = correct_reponses1[correct_reponses1.mean(-1).mean(-1) >= threshold].mean(-1).mean(0)
std_performance1 = correct_reponses1[correct_reponses1.mean(-1).mean(-1) >= threshold].mean(-1).std(0)

mean_performance2 = correct_reponses2[correct_reponses2.mean(-1).mean(-1) >= threshold].mean(-1).mean(0)
std_performance2 = correct_reponses2[correct_reponses2.mean(-1).mean(-1) >= threshold].mean(-1).std(0)

if args.bw:
    colors = ['black', 'black']
    styles = ['-', ':']
    markers = [',',',']
else:
    colors = ['#1f77b4', '#ff7f0e']
    styles = ['-', '-']
    markers = [',',',']

plt.plot(np.arange(0, 10), mean_performance1, c=colors[0], alpha=0.7, marker=markers[0], ls=styles[0])
plt.fill_between(np.arange(0, 10), mean_performance1-(std_performance1 / math.sqrt(correct_reponses1.shape[0])), mean_performance1+(std_performance1 / math.sqrt(correct_reponses1.shape[0])), color=colors[0], alpha=0.1)

plt.plot(np.arange(0, 10), mean_performance2, c=colors[1], alpha=0.7, marker=markers[1], ls=styles[1])
plt.fill_between(np.arange(0, 10), mean_performance2-(std_performance2 / math.sqrt(correct_reponses1.shape[0])), mean_performance2+(std_performance2 / math.sqrt(correct_reponses1.shape[0])), color=colors[1], alpha=0.1)

plt.ylim(0.5, 1)
plt.xlim(0, 9)
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.legend(['First Half', 'Second Half'], frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("plots/figures/performance_half" + str(args.study) + ".pdf", bbox_inches='tight')
plt.show()
