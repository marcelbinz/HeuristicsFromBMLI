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

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
parser.add_argument('--alternative', action='store_true', default=False, help='alternative plotting')
parser.add_argument('--num_experiments', type=int, default=1000, metavar='S', help='number of experiments to run')
parser.add_argument('--num_episodes', type=int, default=30, metavar='S', help='number of episodes per experiment')
parser.add_argument('--sequence_length', type=int, default=10, metavar='S', help='sequence length')
args = parser.parse_args()

if args.recompute:
    if args.study == 1:
        ranking = True
        direction = False

        file_names = [
            'trained_models/ranking_1_0.pth',
            'trained_models/ranking_2_0.pth',
            'trained_models/ranking_3_0.pth',
            'trained_models/ranking_4_0.pth',
            'trained_models/ranking_5_0.pth',
            'trained_models/ranking_6_0.pth',
            'trained_models/ranking_pretrained_0.pth'
            ]

        save_path = 'data/bmli_ranking.pth'

    elif args.study == 2:
        ranking = False
        direction = True

        file_names = [
            'trained_models/direction_1_0.pth',
            'trained_models/direction_2_0.pth',
            'trained_models/direction_3_0.pth',
            'trained_models/direction_4_0.pth',
            'trained_models/direction_5_0.pth',
            'trained_models/direction_6_0.pth',
            'trained_models/direction_pretrained_0.pth'
            ]

        save_path = 'data/bmli_direction.pth'

    elif args.study == 3:
        ranking = False
        direction = False

        file_names = [
            'trained_models/none_1_0.pth',
            'trained_models/none_2_0.pth',
            'trained_models/none_3_0.pth',
            'trained_models/none_4_0.pth',
            'trained_models/none_5_0.pth',
            'trained_models/none_6_0.pth',
            'trained_models/none_pretrained_0.pth'
            ]

        save_path = 'data/bmli_none.pth'

    data_loader = PairedComparison(4, ranking=ranking, direction=direction, dichotomized=False)

    gini_coefficients = torch.zeros(len(file_names), args.num_experiments, args.num_episodes, args.sequence_length)
    map_performance = torch.zeros(len(file_names), args.num_experiments, args.num_episodes, args.sequence_length)
    avg_performance = torch.zeros(len(file_names), args.num_experiments, args.num_episodes, args.sequence_length)

    for m, file_name in enumerate(file_names):
        model = GRU(data_loader.num_inputs, data_loader.num_targets, 128)

        params, _ = torch.load(file_name, map_location='cpu')
        model.load_state_dict(params)

        for k in tqdm(range(args.num_experiments)):
            for i in range(args.num_episodes):
                inputs, targets, _, _ = data_loader.get_batch(1, args.sequence_length)
                predictive_distribution, weights, variances = model(inputs, targets)

                map_performance[m, k, i] = ((predictive_distribution.probs > 0.5).float() == targets).squeeze().detach()
                avg_performance[m, k, i] = ((1 - targets) * (1 - predictive_distribution.probs) + targets * predictive_distribution.probs).squeeze().detach()

                for j in range(args.sequence_length):
                    gini_coefficients[m, k, i, j] = gini(torch.abs(weights[j].t()).squeeze().detach().cpu().numpy())

    print(map_performance.mean(1).mean(1))
    torch.save([map_performance, avg_performance, gini_coefficients], save_path)

else:
    mpl.rcParams['figure.figsize'] = (1.85, 2.25)

    if args.study == 1:
        load_path = 'data/bmli_ranking.pth'
        save_path = 'plots/figures/gini_bmli1_'

    elif args.study == 2:
        load_path = 'data/bmli_direction.pth'
        save_path = 'plots/figures/gini_bmli2_'

    elif args.study == 3:
        load_path = 'data/bmli_none.pth'
        save_path = 'plots/figures/gini_bmli3_'

    map_performance, avg_performance, gini_coefficients = torch.load(load_path)
    print(map_performance.mean(1).mean(1))
    print(avg_performance.mean(1).mean(1))
    if args.alternative:
        time_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        gini_coefficients = gini_coefficients[:, :1, :, time_indices]
        gini_coefficients = gini_coefficients.transpose(1, 3)
        gini_coefficients = gini_coefficients.reshape(gini_coefficients.shape[0], gini_coefficients.shape[1], -1)
        gini_coefficients = gini_coefficients.reshape(gini_coefficients.shape[0], -1)
        cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#1f77b4', '#ff7f0e'])

        for m in range(gini_coefficients.shape[0]):
            print(m)
            plt.figure(m + 1)
            print(gini_coefficients[m].shape)
            t = torch.ones_like(gini_coefficients.flatten()).numpy() #np.concatenate([i * np.ones(gini_coefficients.shape[-1], dtype='int') for i in time_indices])
            g = gini_coefficients.flatten().numpy()#np.concatenate([gini_coefficients[m, i, :].numpy() for i in range(len(time_indices))])
            norm = mpl.colors.Normalize(vmin=0, vmax=0.75)
            colors = {}
            for cval in g:
                colors.update({cval : cmap(norm(cval))})

            df = pd.DataFrame({ 'time step': t, 'Gini coefficients': g})
            sns.swarmplot(x="time step", y="Gini coefficients", hue='Gini coefficients', palette=colors, data=df, alpha=0.7, size=4)
            #plt.clim(0,0.75)
            plt.ylim([-0.08, 0.8])
            plt.xlabel('Time step') #
            plt.ylabel('Gini coefficients')
            plt.axhline(y=0, linewidth=2, color='#1f77b4', ls='--', alpha=0.7)
            plt.axhline(y=0.75, linewidth=2,  color='#ff7f0e', ls='--', alpha=0.7)
            plt.text(5.5, -0.060, 'equal weighting', color='#1f77b4', horizontalalignment='right', alpha=0.7)
            plt.text(5.5, 0.763, 'single cue', color='#ff7f0e', horizontalalignment='right', alpha=0.7)
            plt.gca().legend_.remove()

            sns.despine()

            plt.savefig(save_path + str(m) + '.pdf', bbox_inches='tight')

        plt.show()

    else:

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
            if m == 1:
                plt.ylabel('Gini coefficients')
            else:
                plt.ylabel('')
                plt.yticks([])
            plt.axhline(y=0, linewidth=2, color='#1f77b4', ls='--', alpha=0.7)
            plt.axhline(y=0.75, linewidth=2,  color='#ff7f0e', ls='--', alpha=0.7)
            plt.text(3.5, -0.060, 'equal weighting', color='#1f77b4', horizontalalignment='right', alpha=0.7)
            plt.text(3.5, 0.763, 'single cue', color='#ff7f0e', horizontalalignment='right', alpha=0.7)
            plt.gca().legend_.remove()

            sns.despine()

            plt.savefig(save_path + str(m) + '.pdf', bbox_inches='tight')

        plt.show()
