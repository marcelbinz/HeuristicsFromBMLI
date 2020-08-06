import torch
import scipy.io as sio
import math

logprobs_baseline = torch.load('logprobs1.pth')[0]
sio.savemat('logprobs1.mat', {'lme':logprobs_baseline[:, 1:].detach().numpy()})

logprobs_baseline = torch.load('logprobs2.pth')[0]
sio.savemat('logprobs2.mat', {'lme':logprobs_baseline[:, 1:].detach().numpy()})

logprobs_baseline = torch.load( 'logprobs3.pth')[0]
logprobs_meta = torch.load('logprobs_meta3.pth')[0]
logprobs_meta = torch.cat([logprobs_baseline[:, [0]], logprobs_meta], dim=-1)
best_logprobs, best_index = torch.max(logprobs_meta, dim=-1)
bic = -0.5 * math.log(300) + best_logprobs
logprobs_baseline = torch.cat([logprobs_baseline, bic.unsqueeze(1)], dim=-1)
sio.savemat('logprobs3.mat', {'lme':logprobs_baseline[:, 1:].detach().numpy()})
