import torch
import pandas as pd
import numpy as np


for j in range(1, 5):
    logprobs = torch.load('data/logprobs' + str(j) + '_bmli.pth')[0]
    best_index = torch.argmax(logprobs, -1)

    betas = torch.zeros(best_index.shape)
    for i in range(betas.shape[0]):
        if best_index[i] == 0:
            betas[i] = 0.1
        if best_index[i] == 1:
            betas[i] = 0.03
        if best_index[i] == 2:
            betas[i] = 0.01
        if best_index[i] == 3:
            betas[i] = 0.003
        if best_index[i] == 4:
            betas[i] = 0.001
        if best_index[i] == 5:
            betas[i] = 0.0003
        if best_index[i] == 6:
            betas[i] = 0.0

    print(best_index)

    x_df = pd.DataFrame(betas.numpy())
    x_df.to_csv('data/betas' + str(j) + '.csv')
