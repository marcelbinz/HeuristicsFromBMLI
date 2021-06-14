import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import math
import torch
import numpy as np
import pandas as pd
import statsmodels

# https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    return 0.5 * rmad

def plot_gini(gc, time_indices):
    gc = gc[:, :1, :, time_indices]
    gc = gc.transpose(1, 3)
    gc = gc.reshape(gc.shape[0], gc.shape[1], -1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#1f77b4', '#ff7f0e'])

    fig, ax = plt.subplots(1, gc.shape[0])
    for m in range(gc.shape[0]):
        plt.subplot(1, gc.shape[0], m + 1)
        t = np.concatenate([i * np.ones(gc.shape[-1], dtype='int') for i in time_indices])
        g = np.concatenate([gc[m, i, :].numpy() for i in range(len(time_indices))])

        norm = mpl.colors.Normalize(vmin=0, vmax=0.75)
        colors = {}
        for cval in g:
            colors.update({cval : cmap(norm(cval))})

        df = pd.DataFrame({ 'time step': t, 'Gini coefficients': g})
        sns.swarmplot(x="time step", y="Gini coefficients", hue='Gini coefficients', palette=colors, data=df, alpha=0.7, size=4)
        plt.ylim([-0.08, 0.8])
        plt.xlabel('Trial') #
        if m == 0:
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

    plt.show()

def plot_performance(performance_baselines, performance_bmi, human_correct):
    with torch.no_grad():
        # Baselines and Humans
        plt.subplot(1, 2, 1)
        performance_baselines = performance_baselines[[0, 2, 1]]

        # average over episodes
        performance_baselines = performance_baselines.numpy().mean(2)

         # average over runs
        means_baselines = performance_baselines.mean(axis=1)

        # human performance
        mean_performance = human_correct.mean(-1).mean(0)
        std_performance = human_correct.mean(-1).std(0)

        print("Mean accuracy for all participants:")
        participant_mean = human_correct.mean(-1).mean(-1)
        print(participant_mean.mean())
        print(participant_mean.std())

        # matplotlib settings
        colors = ['#d62728', '#ff7f0e', '#1f77b4', 'black', 'black']
        styles = ['-','-','-','-','-']
        markers = ['o','^','s','*',',']

        for i in range(means_baselines.shape[0]):
            plt.plot(np.arange(1, 11), means_baselines[i], c=colors[i], alpha=0.7, ls=styles[i], marker=markers[i])

        plt.plot(np.arange(1, 11), mean_performance, c=colors[4], alpha=0.7, ls=styles[4], marker=markers[4])
        plt.fill_between(np.arange(1, 11), mean_performance-(std_performance / math.sqrt(human_correct.shape[0])), mean_performance+(std_performance  / math.sqrt(human_correct.shape[0])), color=colors[4], alpha=0.1)

        plt.legend(['Ideal Observer', 'Single Cue', 'Equal Weighting', 'Humans'], frameon=False, loc=(0, 0.65), fontsize=8)
        plt.ylim(0.5, 1)
        plt.xlim(1, 10)
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        sns.despine()
        plt.tight_layout()

        # BMI
        plt.subplot(1, 2, 2)
        means_bmi = performance_bmi.mean(1).mean(1)[[0, 2, 4, 6]]
        styles = [(0, (5, 4)), (0, (5, 3)), (0, (5, 1)),'-' ]

        for i in range(means_bmi.shape[0]):
            plt.plot(np.arange(1, 11), means_bmi[i], c='black', alpha=0.7, ls=styles[i])

        plt.legend(['BMI ' + r'($\beta = 0.1$)','BMI ' + r'($\beta = 0.01$)', 'BMI ' + r'($\beta = 0.001$)', 'MI'], frameon=False,  loc=(0, 0.65), fontsize=8) #
        plt.ylim(0.5, 1)
        plt.xlim(1, 10)
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        sns.despine()
        plt.tight_layout()

        plt.show()

def plot_performance_ffn(performance):
    means = performance.mean(1).mean(1)[[0,1,3,5]]

    styles = [(0, (5, 4)), (0, (5, 3)), (0, (5, 1)),'-' ]
    for i in range(means.shape[0]):
        plt.plot(np.arange(1, 11), means[i], c='black', alpha=0.7, ls=styles[i])

    plt.legend([ r'$\alpha = 0$', r'$\alpha = 2^{-8}$', r'$\alpha = 2^{-6}$', r'$\alpha = 2^{-4}$'], frameon=False,  loc=(0, 0.6), fontsize=8)
    plt.ylim(0.5, 1)
    plt.xlim(1, 10)
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_strategy_selection(performance, selected_models):
    means = selected_models.mean(0).mean(0)

    colors = ['#d62728', '#1f77b4', '#ff7f0e']
    styles = ['-','-','-',]
    markers = ['o','s', '^']

    plt.figure(1)
    performance = performance.detach().numpy().mean(2)
    means_ss = performance.mean(axis=1)
    for i in range(means_ss.shape[0]):
        plt.plot(np.arange(1, 11), means_ss[i], color='black', alpha=0.7)
    plt.ylim(0.5, 1)
    plt.xlim(1, 10)
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    sns.despine()
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    for i in range(means.shape[1]):
        plt.plot(torch.arange(1, 11), means[:, i], c=colors[i], alpha=0.7, ls=styles[i], marker=markers[i])

    plt.ylim(0.0, 1)
    plt.xlim(1, 10)
    plt.legend(['Ideal Observer', 'Equal Weighting', 'Single Cue'], frameon=False,  loc=(0, 0.735), fontsize=8)
    plt.xlabel('Trial')
    plt.ylabel('Strategy Percentage')
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_comparison(model_index, logprobs_baselines, logprobs_selection, logprobs_feedforward, logprobs_bmi=None, t_test=False):
    logprobs_baselines[:, 1:] = -0.5 * math.log(300) + logprobs_baselines[:, 1:]
    logprobs_selection = -0.5 * math.log(300) + logprobs_selection
    logprobs_feedforward = -0.5 * 2 * math.log(300) + logprobs_feedforward

    if logprobs_bmi is not None:
        logprobs_bmi = torch.cat([logprobs_baselines[:, [0]], logprobs_bmi], dim=-1)
        best_logprobs, best_index = torch.max(logprobs_bmi, dim=-1)
        bic = -0.5  * math.log(300) + best_logprobs
        logprobs = torch.cat([logprobs_baselines, logprobs_selection, logprobs_feedforward.unsqueeze(1), bic.unsqueeze(1)], dim=-1)
        ylabels = ['Guessing', 'Ideal Observer', 'Equal Weighting', 'Single Cue', 'Strategy Selection', 'Feedforward Network', 'BMI']
    else:
        logprobs = torch.cat([logprobs_baselines, logprobs_selection, logprobs_feedforward.unsqueeze(1)], dim=-1)
        ylabels = ['Guessing', 'Ideal Observer', 'Equal Weighting', 'Single Cue', 'Strategy Selection', 'Feedforward Network']

    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

    print('Number of participants best explained by hypothesis:')
    print((torch.argmax(posterior_logprob.exp().detach(), dim=-1) == model_index).sum())

    print('Number of participants better explained than 0.99:')
    print((posterior_logprob.exp().detach()[:, model_index] > 0.99).sum())

    all_joint_logprob = joint_logprob.sum(0, keepdim=True)
    all_marginal_logprob = torch.logsumexp(all_joint_logprob, dim=1, keepdim=True)
    all_posterior_logprob = all_joint_logprob - all_marginal_logprob
    print('Joint posterior for each model: ' + str(all_posterior_logprob.exp()))

    if t_test:
        betas = torch.Tensor([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0])
        betas_bmi = betas[best_index[posterior_logprob.exp().detach().argmax(-1) == 6]]
        betas_other = betas[best_index[posterior_logprob.exp().detach().argmax(-1) != 6]]
        tstat, pvalue, df = statsmodels.stats.weightstats.ttest_ind(betas_bmi.detach().numpy(),betas_other.detach().numpy(), alternative='smaller', usevar='unequal')
        print('Participants best described by BMI have lower resource limitations:')
        print('t(' + str(df) + ') = ' + str(tstat) + ', p = ' + str(pvalue))

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(cbar_ax)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#ffffff', '#363737'])
    ax = sns.heatmap(posterior_logprob.exp().t().detach(), cmap=cmap, vmax=1.0, center=0.5, square=True, linewidths=.5, ax=ax, cbar_ax=cbar_ax, linecolor='#d8dcd6')
    fig.axes[-1].yaxis.set_ticks([0, 1.0])

    ax.set_yticklabels(ylabels, rotation='horizontal')
    ax.set_xlabel('Participant')
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 5 != 0]
    plt.tight_layout()
    plt.show()

def plot_comparison_time(logprobs_time, model_index):
    models = [1, 2, 3, 4]
    models.remove(model_index)
    for i, model in enumerate(models):
        plt.subplot(1, 3, i+1)
        y = -(logprobs_time[model_index] - logprobs_time[model]).sum(-1).t().detach()
        plt.plot(np.arange(1, 11), y.mean(-1), color='C0', alpha=0.8)
        plt.errorbar(np.arange(1, 11),y.mean(-1), yerr=y.std(-1)/ math.sqrt(y.shape[1]) )
        plt.axhline(y=0.0, color='black', linestyle='--')
        plt.plot(np.arange(1, 11), y, color='C0', alpha=0.1)
        plt.ylim(-10, 10)
        plt.xlim(1, 10)
        if i == 0:
            plt.ylabel(r'$\Delta \log p(\hat{\mathbf{c}}^{(i)}| \mathbf{X}^{(i)}, m, \theta^*)$')
        else:
            plt.ylabel('')
            plt.yticks([])
        plt.xlabel('Trial')
        sns.despine()
    plt.show()

def plot_power(kl):
    mean = kl.sum(-1).mean(1)
    variance = kl.sum(-1).std(1).pow(2)
    num_tasks = torch.arange(1, 31)
    expected_bf = torch.arange(1, 31).unsqueeze(-1) * mean
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
    plt.show()

def plot_kl(kl_divergences):
    with torch.no_grad():
        max_kl = torch.max(kl_divergences[~torch.isinf(kl_divergences)])
        kl_divergences[torch.isinf(kl_divergences)] = max_kl
        total_kl = kl_divergences.sum(-1).mean(-1).mean(-1)
        for i in range(1):
            plt.figure(i)
            plt.xlabel('KL Divergence')
            plt.text(0.1, -0.1, 'equal weighting', color='black', horizontalalignment='left')
            plt.text(0.1, 0.9, 'single cue', color='black', horizontalalignment='left')
            plt.barh([0, 1], total_kl[i], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
            sns.despine()
            plt.ylabel('')
            plt.yticks([])
            plt.tight_layout()
        plt.show()

def plot_tasks(baselines_logprob, selection_logprob, feedforward_logprob):

    baselines_logprob[:, :, 1:] = baselines_logprob[:, :, 1:] - 0.5 * math.log(10)
    baselines_logprob = baselines_logprob.sum(1)


    selection_logprob = selection_logprob - 0.5 * math.log(10)
    selection_logprob = selection_logprob.sum(1)

    feedforward_logprob = feedforward_logprob - 0.5 * 2 * math.log(10)
    feedforward_logprob = feedforward_logprob.sum(1, keepdim=True)

    logprobs = torch.cat([baselines_logprob, selection_logprob, feedforward_logprob], dim=-1)

    ylabels = ['Guessing', 'Ideal Observer', 'Equal Weighting', 'Single Cue', 'Strategy Selection', 'Feedforward Network']


    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

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
    plt.show()

def plot_performance_half(targets, predictions):
    correct_reponses = (targets == predictions).view(targets.shape[0],targets.shape[1], -1).float()
    correct_reponses1 = correct_reponses[:, :, :15]
    correct_reponses2 = correct_reponses[:, :, 15:]

    mean_performance1 = correct_reponses1.mean(-1).mean(0)
    std_performance1 = correct_reponses1.mean(-1).std(0)

    mean_performance2 = correct_reponses2.mean(-1).mean(0)
    std_performance2 = correct_reponses2.mean(-1).std(0)

    colors = ['black', 'black']
    styles = ['-', ':']
    markers = [',',',']

    plt.plot(np.arange(1, 11), mean_performance1, c=colors[0], alpha=0.7, marker=markers[0], ls=styles[0])
    plt.fill_between(np.arange(1, 11), mean_performance1-(std_performance1 / math.sqrt(correct_reponses1.shape[0])), mean_performance1+(std_performance1 / math.sqrt(correct_reponses1.shape[0])), color=colors[0], alpha=0.1)

    plt.plot(np.arange(1, 11), mean_performance2, c=colors[1], alpha=0.7, marker=markers[1], ls=styles[1])
    plt.fill_between(np.arange(1, 11), mean_performance2-(std_performance2 / math.sqrt(correct_reponses1.shape[0])), mean_performance2+(std_performance2 / math.sqrt(correct_reponses1.shape[0])), color=colors[1], alpha=0.1)

    plt.ylim(0.5, 1)
    plt.xlim(1, 10)
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.legend(['First Half', 'Second Half'], frameon=False, loc=(0, 0.835), fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.show()
