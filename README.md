# Heuristics From Bounded Meta-Learned Inference

<b> PsyArXiv </b>: TODO

<b> Abstract </b>: Numerous researchers have put forward heuristics as models of human decision making. However, where such heuristics come from is still a topic of ongoing debates. In this work we propose a novel computational model that advances our understanding of heuristic decision making by explaining how different heuristics are discovered and how they are selected. This model, called bounded meta-learned inference, is based on the idea that people make environment-specific inferences about which strategies to use, while being efficient in terms of how they use computational resources. We show that our approach discovers two previously suggested types of heuristics -- one reason decision making and equal weighting -- in specific environments. Furthermore, the model provides clear and precise predictions about when each heuristic should be applied: knowing the correct ranking of attributes leads to one reason decision making, knowing the directions of the attributes leads to equal weighting, and not knowing about either leads to strategies that use weighted combinations of multiple attributes. This allows us to gain new insights on mixed results of prior empirical work on heuristic decision making. In three empirical paired comparison studies with continuous features, we verify predictions of our theory, and show that it captures several characteristics of human decision making not explained by alternative theories.

## Install

```console
git clone https://github.com/marcelbinz/HeuristicsFromBMLI.git
cd HeuristicsFromBMLI
pip install -e .
```

## Empirical data description
```console
import torch
load_path = "data/humans_ranking.pth"
inputs_a, inputs_b, targets, predictions, _ = torch.load(load_path)

inputs_a.shape
torch.Size([28, 10, 30, 4])
inputs_b.shape
torch.Size([28, 10, 30, 4])
targets.shape
torch.Size([28, 10, 30, 1])
predictions.shape
torch.Size([28, 10, 30, 1])
```

Dimension | Contents
--- | --- 
1 | Participants
2 | Trials
3 | Tasks
4 | Features

For experiments 2 and 3 use load_path = "data/humans_direction.pth" and load_path = "data/humans_none.pth" respectively.

## Create all plots (in order of appearance)

```console
python plots/gini_bmli.py --study 1		(Figure 3 (a) & (b))
python plots/gini_bmli.py --study 2		(Figure 4 (a) & (b))
python plots/gini_bmli.py --study 3		(Figure 5 (a) & (b))

python plots/gini_ideal.py --study 1		(Figure 3 (c))
python plots/gini_ideal.py --study 2		(Figure 4 (c))
python plots/gini_ideal.py --study 3		(Figure 5 (c))

python plots/performance_all.py --study 1	(Figure 7 (a))
python plots/performance_all.py --study 2	(Figure 8 (a))
python plots/performance_all.py --study 3	(Figure 9 (a))

python plots/comparison_baselines.py --study 1	(Figure 7 (b))
python plots/comparison_baselines.py --study 2	(Figure 8 (b))
python plots/comparison_baselines.py --study 3	(Figure 9 (b))

python plots/performance_half.py --study 1	(Figure 10 (a))
python plots/performance_half.py --study 2	(Figure 10 (b))
python plots/comparison_tasks.py --study 1	(Figure 10 (c))
python plots/comparison_tasks.py --study 2 	(Figure 10 (d))

python plots/performance_bmli.py --study 3	(Figure 11)

python plots/power_analysis.py			(Figure C1)
```

## Run additional model comparisons

```console
python plots/comparison_bmli.py --study 1
python plots/comparison_bmli.py --study 2
python plots/comparison_bmli.py --study 3
```

## Train all BMLI models (optional, pretrained models already included)

```console
./experiments/experiment_ranking.sh
./experiments/experiment_direction.sh
./experiments/experiment_none.sh
```

## Run all model simulations (optional, model simulation results already included)

```console
python models/baselines.py

python plots/gini_bmli.py --study 1 --recompute
python plots/gini_bmli.py --study 2 --recompute
python plots/gini_bmli.py --study 3 --recompute

python plots/comparison_baselines.py --study 1 --recompute
python plots/comparison_baselines.py --study 2 --recompute
python plots/comparison_baselines.py --study 3 --recompute

python plots/comparison_bmli.py --study 1 --recompute
python plots/comparison_bmli.py --study 2 --recompute
python plots/comparison_bmli.py --study 3 --recompute

python plots/comparison_tasks.py --study 1 --recompute
python plots/comparison_tasks.py --study 2 --recompute

python plots/power_analysis.py --recompute
```
