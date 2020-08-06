import torch
from models.model import GRU
from models.baselines import Guessing, VariationalProbitRegression, VariationalEqualWeighting, VariationalFirstCue, VariationalBestCue
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import argparse
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
args = parser.parse_args()

if args.recompute:
    if args.study == 1:
        load_path = "data/humans_ranking.pth"
    elif args.study == 2:
        load_path = "data/humans_direction.pth"
    elif args.study == 3:
        load_path = "data/humans_none.pth"

    save_path = 'data/logprobs' + str(args.study) + '_tasks.pth'

    inputs_a, inputs_b, targets, predictions, _ = torch.load(load_path)

    if args.study == 1:
        models = [Guessing, VariationalProbitRegression, VariationalEqualWeighting, VariationalFirstCue]
    else:
        models = [Guessing, VariationalProbitRegression, VariationalEqualWeighting, VariationalBestCue]

    logprobs = torch.zeros(inputs_a.shape[2], len(models))
    for participant in tqdm(range(inputs_a.shape[0])):
        for m, model_class in enumerate(models):
            for task in range(inputs_a.shape[2]):
                model = model_class(4)

                participant_inputs = inputs_a[participant, :, [task]] - inputs_b[participant, :, [task]]
                participant_targets = targets[participant, :, [task]]
                predictive_distribution = model.forward(participant_inputs, participant_targets)
                logprobs[task, m] = logprobs[task, m] + predictive_distribution.log_prob(predictions[participant, :, [task]]).sum()

    print(logprobs / 2.303) #
    print(torch.argmax(logprobs, -1))

    torch.save([logprobs], save_path)

else:
    load_path = 'data/logprobs' + str(args.study) + '_tasks.pth'
    logprobs = torch.load(load_path)[0]

    ylabels = ['Guessing', 'Ideal Observer', 'Equal Weighting', 'Single Cue']


    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob
    print(posterior_logprob.exp())

    mpl.rcParams['figure.figsize'] = (6.5, 2.5)
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(cbar_ax)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#ffffff', '#363737'])
    ax = sns.heatmap(posterior_logprob.exp().t().detach(), cmap=cmap, vmax=1.0, center=0.5, square=True, linewidths=.5, ax=ax, cbar_ax=cbar_ax, linecolor='#d8dcd6')
    fig.axes[-1].yaxis.set_ticks([0, 1.0])

    ax.set_yticklabels(ylabels, rotation='horizontal')
    ax.set_xlabel('Task')
    n = 5  # Keeps every 5th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    plt.tight_layout()
    fig.savefig('plots/posterior_tasks' + str(args.study) + '.pdf', bbox_inches='tight')
    plt.show()
