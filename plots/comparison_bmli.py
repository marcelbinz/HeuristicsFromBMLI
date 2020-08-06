import torch
from model import GRU
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import argparse
from torch.distributions import Bernoulli
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--study', type=int, default=1, metavar='S', help='study')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
parser.add_argument('--samples', type=int, default=100, metavar='NS', help='number of samples')
args = parser.parse_args()

if args.recompute:
    if args.study == 1:
        load_path = "data/humans_ranking.pth"
    elif args.study == 2:
        load_path = "data/humans_direction.pth"
    elif args.study == 3:
        load_path = "data/humans_none.pth"
    save_path = 'data/logprobs' + str(args.study) + '_bmli.pth'

    inputs_a, inputs_b, targets, predictions, _ = torch.load(load_path)

    if args.study == 1:
        model_paths = [
            'trained_models/ranking_1_0.pth',
            'trained_models/ranking_2_0.pth',
            'trained_models/ranking_3_0.pth',
            'trained_models/ranking_4_0.pth',
            'trained_models/ranking_5_0.pth',
            'trained_models/ranking_6_0.pth',
            'trained_models/ranking_pretrained_0.pth'
            ]
    elif args.study == 2:
        model_paths = [
            'trained_models/direction_1_0.pth',
            'trained_models/direction_2_0.pth',
            'trained_models/direction_3_0.pth',
            'trained_models/direction_4_0.pth',
            'trained_models/direction_5_0.pth',
            'trained_models/direction_6_0.pth',
            'trained_models/direction_pretrained_0.pth'
            ]

    elif args.study == 3:
        model_paths = [
            'trained_models/none_1_0.pth',
            'trained_models/none_2_0.pth',
            'trained_models/none_3_0.pth',
            'trained_models/none_4_0.pth',
            'trained_models/none_5_0.pth',
            'trained_models/none_6_0.pth',
            'trained_models/none_pretrained_0.pth'
            ]

    logprobs = torch.zeros(inputs_a.shape[0], len(model_paths))
    # for each participant
    with torch.no_grad():
        for participant in tqdm(range(inputs_a.shape[0])):
            # for each model
            for m, model_path in enumerate(model_paths):
                model = GRU(4, 1, 128)

                params, _ = torch.load(model_path, map_location='cpu')
                model.load_state_dict(params)

                participant_inputs = inputs_a[participant] - inputs_b[participant]
                participant_targets = targets[participant]

                avg_probs = 0
                for sample in range(args.samples):
                    predictive_distribution, _, _ = model(participant_inputs, participant_targets)
                    avg_probs += predictive_distribution.probs

                avg_predictive_distribution = Bernoulli(avg_probs / args.samples)
                logprobs[participant, m] = avg_predictive_distribution.log_prob(predictions[participant]).sum()

        print(logprobs / 2.303)
        print(torch.argmax(logprobs, -1))
        torch.save([logprobs], save_path)

else:
    logprobs_baseline = torch.load('data/logprobs' + str(args.study) + '.pth')[0]
    logprobs_meta = torch.load('data/logprobs' + str(args.study) + '_bmli.pth')[0]
    print(logprobs_meta.shape)

    logprobs_meta = torch.cat([logprobs_baseline[:, [0]], logprobs_meta], dim=-1)
    best_logprobs, best_index = torch.max(logprobs_meta, dim=-1)

    bic = -0.5 * math.log(300) + best_logprobs

    logprobs = torch.cat([logprobs_baseline, bic.unsqueeze(1)], dim=-1).sum(0)
    print(logprobs)

    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[0])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=0, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

    print(posterior_logprob.exp())
