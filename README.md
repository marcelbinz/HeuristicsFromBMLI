# Heuristics From Bounded Meta-Learned Inference

<b> PsyArXiv</b>: [Heuristics From Bounded Meta-Learned Inference](https://psyarxiv.com/5du2b/)

<b> Abstract</b>: Numerous researchers have put forward heuristics as models of human decision-making. However, where such heuristics come from is still a topic of ongoing debate. In this work, we propose a novel computational model that advances our understanding of heuristic decision-making by explaining how different heuristics are discovered and how they are selected. This model -- called bounded meta-learned inference (BMI) -- is based on the idea that people make environment-specific inferences about which strategies to use while being efficient in terms of how they use computational resources. We show that our approach discovers two previously suggested types of heuristics -- one reason decision-making and equal weighting -- in specific environments. Furthermore, the model provides clear and precise predictions about when each heuristic should be applied: knowing the correct ranking of attributes leads to one reason decision-making, knowing the directions of the attributes leads to equal weighting, and not knowing about either leads to strategies that use weighted combinations of multiple attributes. This allows us to gain new insights on mixed results of prior empirical work on heuristic decision-making. In three empirical paired comparison studies with continuous features, we verify predictions of our theory and show that it captures several characteristics of human decision-making not explained by alternative theories.

## Empirical data description

Data for all studies is available as .csv files.

Experiment | Filename
--- | --- 
1 | exp1.csv
2 | exp2.csv
3 | exp4.csv
3b | exp3.csv

Each .csv file contains 10 columns:

* `participant`: unique participant id
* `task`: unique task id
* `step`: unique time-step id
* `x0`: difference in first feature
* `x1`: difference in second feature
* `x2`: difference in third feature
* `x3`: difference in fourth feature
* `choice`: participant choice
* `target`: correct choice
* `time`: total time passed (in milliseconds)

Alternatively data is also available in pytorch format:
```console
import torch

# Experiment 1
load_path = "data/humans_ranking.pth"
inputs_a, inputs_b, targets, choices, time_elapsed = torch.load(load_path)

inputs_a.shape
torch.Size([28, 10, 30, 4])
inputs_b.shape
torch.Size([28, 10, 30, 4])
targets.shape
torch.Size([28, 10, 30, 1])
choices.shape
torch.Size([28, 10, 30, 1])
```

Dimension | Contents
--- | --- 
1 | Participants
2 | Trials
3 | Tasks
4 | Features

For experiments 2, 3 and 3b use: 

```console
import torch

# Experiment 2
load_path = "data/humans_direction.pth"
inputs_a, inputs_b, targets, choices, time_elapsed = torch.load(load_path)

# Experiment 3
load_path = "data/humans_2features.pth"
inputs_a, inputs_b, targets, choices, time_elapsed, weights, direction1, direction2, ranking = torch.load(load_path)

# Experiment 3b
load_path = "data/humans_none.pth"
inputs_a, inputs_b, targets, choices, time_elapsed = torch.load(load_path)

```

## Documentation

Filename | Contents
--- | --- 
plots.ipynb | Replicate all plots
stats.html | Additional statistical analysis
eval.ipynb | Evaluate all models (optional, model simulation results already included) 
train.py | Train all BMI models (optional, pretrained models already included)
utils.py | Helper functions for plotting and loading
model.py | Implementation of BMI
baselines.py | Implementations of all other models
environments.py | Data generation for paired comparison task
