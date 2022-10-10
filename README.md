# SCMvL: A Search-based Contrastive Multi-View Learning for Graph Classification.
The original implementation for paper SCMvL: A Search-based Contrastive Multi-View Learning for Graph Classification.

SCMvL approach systematically composes multi-view from a sample graph engaging a pre-trained framework using contrastive learning and achieves state-of-the-art on 7 out of 10 datasets for graph classification task.

## Installation
### Requirements

```
Linux with Python ≥ 3.6
PyTorch ≥ 1.4.0
tensorboard
setuptools==59.5.0
```

## Test
For testing purposes, you can run this script. `python classification.py --load pro_ptrain_gcn --logdir pro_ftune_gcn --dataset proteins --model gcn`. 
We have provided a pre-train embedding on the protein dataset for testing purposes to prevent one from going through the prolonged training time. 

For reproducibility, follow the instructions below.  

## Pre-training
For pre-training task, run:
`python execute.py --logdir pro_ptrain_gcn --dataset proteins --model gcn --augment_list edge_perturbation  node_attr_mask`

The model options are gcn, graphsage and gin.

Dataset: nci1, proteins, dd, enzymes, mutag, collab, imdb_multi, imdb_binary, reddit_multi and reddit_binary.

Augmentation choices are "node_dropping", "node_attr_mask", "edge_perturbation", "diffusion" and  "random_walk_subgraph".

## Downstream Tasks
For downstrean task, run:
`python classification.py --load pro_ptrain_gcn --logdir pro_ftune_gcn --dataset proteins --model gcn`

### Graph Classification Results
Overall, attribute masking performs across various dataset. The result for all biochemical molecular graph network datasets is a combination of random walk and node attribute masking for view generation. We perform 10 runs to evaluate the model’s performance and report the mean accuracy and standard deviation. The best performances are highlighted in bold and italicized as the second best.
|          | NCI1       | PROTEINS   | D&D        | ENZYMES    | MUTAG      |
| :---     | :---:      | :---:      | :---:      |   :---:    |     :---:  |
| GCN      | 69.8 ± 2.2 | 72.8 ± 3.7 | 71.4 ± 4.5 | 65.2 ± 6.4 | 85.6 ± 5.8 | 
| DGCNN    | 76.4 ± 1.7 | 72.9 ± 3.5 | 76.6 ± 4.3 | 38.9 ± 5.7 | -          | 
| DiffPool | 76.9 ± 1.9 | 73.7 ± 3.5 | 75.0 ± 3.5 | 59.5 ± 5.6 | -          |
| ECC      | 76.2 ± 1.4 | 72.3 ± 3.4 | 72.6 ± 4.1 | 29.5 ± 8.2 | -          |
| GIN      | *80.0 ± 1.4* | 73.3 ± 4.0 | 75.3 ± 2.9 | 59.6 ± 4.5 | -|
| GraphSAGE| 76.0 ± 1.8 | 73.0 ± 4.5 | 72.9 ± 2.0 | 58.2 ± 6.0 | - |
| InfoGraph| 76.2 ± 1.1 | 74.44 ± 0.3 | 72.85 ± 1.8 | - | 89.01 ± 1.1 |
| GraphCL  | 77.87 ± 0.4 | 74.39 ± 0.5 | 78.62 ± 0.4 | - | 86.90 ± 1.3|
| GCC      | -           | -           | -           | - | -          |
| MVGRL    | -           | -           | -           | -  | *89.7 ± 1.1* |
| JOAO     | 78.07 ± 0.47 | *74.55 ± 0.41* | 77.40 ± 1.15 | - | 87.67 ± 0.79 |
| Mixup    | **81.0 ± 1.9** | 74.3 ± 3.5 | 76.8 ± 2.9 | - | - |
| ***SCMvL+GCN***| 72.39 ± 0.16 | 73.38 ± 0.32 | *82.43 ± 0.25* | *82.29 ± 0.52* | 87.02 ± 0.04 |
| ***SCMvL+GraphSAGE*** | 74.95 ± 0.12 | **76.89 ± 0.36** | **85.63 ± 1.5**| **94.00 ± 0.29** | **89.76 ± 0.52** |
| ***SCMvL+GIN*** | 71.45 ± 0.18 | 72.26 ± 0.21 | 76.33 ± 0.26 | 57.10 ± 0.48 | 84.22 ± 0.51 |

The result for all social graph network datasets is a combination of edge perturbation and node attribute masking for view generation.
|          | COLLAB     |IMDB-M       | IMDB-B      | REDDIT-M5K  | REDDIT-B   |
| :---     | :---:      | :---:       | :---:       |   :---:     |     :---:  |
| GCN      | 70.2 ± 1.5 |*51.9 ± 3.8* | 74.0 ± 3.4  | 52.2 ± 1.5  | 50.0 ± 0.0 |
| DGCNN    | 71.2 ± 1.9 | 45.6 ± 3.4  | 69.2 ± 3.0  | 49.2 ± 1.2  | 87.8 ± 2.5 |
| DiffPool | 68.9 ± 2.0 | 45.6 ± 3.4  | 68.4 ± 3.3  | 53.8 ± 1.4  | 89.1 ± 1.6 |
| ECC      | OOM        | 43.5 ± 3.1  | 67.7 ± 2.8  | OOM         | OOM        |
| GIN      | 75.6 ± 2.3 | 48.5 ± 3.3  | 71.2 ± 3.9  | 56.1 ± 1.7  | *89.9 ± 1.9*|
| GraphSAGE| 73.9 ± 1.7 | 47.6 ± 3.5  | 68.8 ± 4.5  | 50.0 ± 1.3  | 84.3 ± 1.9  |
| InfoGraph| 70.65 ± 1.1 | -          | 73.0 ± 0.8  | 53.46 ± 1.0 | 82.5 ± 1.4 |
| GraphCL  | 71.4 ± 1.1 | -           | 71.14 ± 0.4 | 55.9 ± 0.3  | 86.90 ± 1.3 |
| GCC      | **81**     | **52**      | 75          | 53.0        | 87.8       |
| MVGRL    | -          | 51.2 ± 0.5  | 74.2 ± 0.7  | -           | 84.5 ± 0.6 |
| JOAO     | 69.33 ± 0.36 | -         | 70.83 ± 0.25 | 56.03 ± 0.27 | 86.42 ± 1.45 |
| Mixup    | 77.0 ± 2.2 | 49.9 ± 3.2  | -            |57.8 ± 1.7  | -      |
|***SCMvL+GCN***| 78.34 ± 0.07 | 50.25 ± 0.17 |**76.41 ± 2.31**| 54.2 ± 0.04| 86.7 ± 0.02 |
|***SCMvL+GraphSAGE*** | **79.27 ± 0.15** | 51.76 ± 0.19 | *76.25 ± 0.93* | 54.2 ± 0.04 | 89.12 ± 2.07 |
|***SCMvL+GIN*** | 75.8 ± 0.06 | 51.59 ± 0.22 | 71.86 ± 3.1 | **57.86 ± 2.3** | **89.97 ± 2.1** |



## Citing SCMvL

## Acknowledgements
Part of this code is inspired by 
1. [PyTorch Geometric](pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)
2. [DIG: Dive into Graphs is a turnkey library for graph deep learning research.](https://github.com/divelab/DIG)
3. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
4. [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
5. [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
6. [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
7. [Self-Supervised Learning For Graphs](https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809)
