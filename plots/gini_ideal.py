from models.environments import PairedComparison
from model import GRU
import torch
import numpy as np
from tqdm import tqdm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pandas as pd

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
args = parser.parse_args()

mpl.rcParams['figure.figsize'] = (1.85, 2.25)

if args.study == 1:
    load_path = 'data/baselines_ranking.pth'
    save_path = 'plots/figures/gini_ideal1'

elif args.study == 2:
    load_path = 'data/baselines_direction.pth'
    save_path = 'plots/figures/gini_ideal2'

elif args.study == 3:
    load_path = 'data/baselines_none.pth'
    save_path = 'plots/figures/gini_ideal3'

map_performance, avg_performance, gini_coefficients = torch.load(load_path)

time_indices = [1, 2, 4, 8]

gini_coefficients = gini_coefficients[:, :1, :, time_indices]
gini_coefficients = gini_coefficients.transpose(1, 3)
gini_coefficients = gini_coefficients.reshape(gini_coefficients.shape[0], gini_coefficients.shape[1], -1)
cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#1f77b4', '#ff7f0e'])

for m in range(gini_coefficients.shape[0]):
    print(m)
    plt.figure(m + 1)
    print(gini_coefficients[m].shape)
    t = np.concatenate([i * np.ones(gini_coefficients.shape[-1], dtype='int') for i in time_indices])
    g = np.concatenate([gini_coefficients[m, i, :].numpy() for i in range(len(time_indices))])

    norm = mpl.colors.Normalize(vmin=0, vmax=0.75)
    colors = {}
    for cval in g:
        colors.update({cval : cmap(norm(cval))})

    df = pd.DataFrame({ 'time step': t, 'Gini coefficients': g})
    sns.swarmplot(x="time step", y="Gini coefficients", hue='Gini coefficients', palette=colors, data=df, alpha=0.7, size=4)
    #plt.clim(0,0.75)
    plt.ylim([-0.08, 0.8])
    plt.xlabel('Trial') #
    plt.yticks([])
    plt.ylabel('')
    plt.axhline(y=0, linewidth=2, color='#1f77b4', ls='--', alpha=0.7)
    plt.axhline(y=0.75, linewidth=2,  color='#ff7f0e', ls='--', alpha=0.7)
    plt.text(3.5, -0.060, 'equal weighting', color='#1f77b4', horizontalalignment='right', alpha=0.7)
    plt.text(3.5, 0.763, 'single cue', color='#ff7f0e', horizontalalignment='right', alpha=0.7)
    plt.gca().legend_.remove()

    sns.despine()

    plt.savefig(save_path + '.pdf', bbox_inches='tight')

plt.show()
