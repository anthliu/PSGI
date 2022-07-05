# Learning Parameterized Task Structure for Generalization to Unseen Entities

Code for an implementation of Learning Parameterized Task Structure for Generalization to Unseen Entities
published at [AAAI 2022](https://www.aaai.org/AAAI22Papers/AAAI-6062.LiuA.pdf).

[arxiv](https://arxiv.org/abs/2203.15034). [project website](https://sites.google.com/umich.edu/psgi/home).
[video intro](https://aaai-2022.virtualchair.net/poster_aaai6062).

## Installation

```shell
conda create -n psgi python=3.8
cd <PSGI directory>
pip install -e .
```

## Scripts

Run the following commands to run the experiments with one of the following environments: `cooking`, `ETmining`, or `ai2thor`.

**Random**
```shell
bash script/run_rl_baselines.sh --algorithm=random --env_id=cooking --graph_param=eval --seed 1
```

**HRL**
```shell
 bash script/run_hrl.sh --env_id=cooking --graph_param=eval --seed 1
```

**MSGI plus**
```shell
 bash script/run_msgi_plus.sh --env_id=cooking --graph_param=eval --seed 1
```

**PSGI (no prior graph)**
```shell
 bash script/run_np_psgi.sh --env_id=cooking --graph_param=eval --seed 1
```

**PSGI**
```shell
 bash script/meta_train_psgi.sh --env_id=cooking --graph_param=train --seed 1 --exp_id 1 # train and save psgi graphs
 bash script/meta_eval_psgi.sh --env_id=cooking --graph_param=eval --seed 1 --exp_id 1 --load_exp_id 1 # eval and load psgi graphs from train
```

## Cite this work

```
@inproceedings{liu2022learning,
  title={Learning Parameterized Task Structure for Generalization to Unseen Entities},
  author={Liu, Anthony and Sohn, Sungryull and Qazwini, Mahdi and Lee, Honglak},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={7},
  pages={7534--7541},
  year={2022}
}
```
