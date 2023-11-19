# CrowdNavRL

**[`Website`](https://www.epfl.ch/labs/vita/research/planning/crowd-robot-interaction/) | [`Paper`](https://arxiv.org/abs/1809.08835) | [`Video`](https://youtu.be/0sNVtQ9eqjA)**

This repository contains the code for the following papers:
- [Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning](https://arxiv.org/abs/1609.07845).
- [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1805.01956).
- [Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).
- [Adaptive Environment Modeling Based Reinforcement Learning for Collision Avoidance in Complex Scenes](https://arxiv.org/abs/2203.07709).

More recent work in the following links 
- [Relational Graph Learning for Crowd Navigation, IROS, 2020](https://github.com/ChanganVR/RelationalGraphLearning).
- [Social NCE: Contrastive Learning of Socially-aware Motion Representations, ICCV, 2021](https://github.com/vita-epfl/social-nce).

## Abstract
Mobility in an effective and socially-compliant manner is an essential yet challenging task for robots operating in crowded spaces.
Recent works have shown the power of deep reinforcement learning techniques to learn socially cooperative policies.
However, their cooperation ability deteriorates as the crowd grows since they typically relax the problem as a one-way Human-Robot interaction problem.
In this work, we want to go beyond first-order Human-Robot interaction and more explicitly model Crowd-Robot Interaction (CRI).
We propose to (i) rethink pairwise interactions with a self-attention mechanism, and
(ii) jointly model Human-Robot as well as Human-Human interactions in the deep reinforcement learning framework.
Our model captures the Human-Human interactions occurring in dense crowds that indirectly affects the robot's anticipation capability.
Our proposed attentive pooling mechanism learns the collective importance of neighboring humans with respect to their future states.
Various experiments demonstrate that our model can anticipate human dynamics and navigate in crowds with time efficiency,
outperforming state-of-the-art methods.


## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  OM-SARL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />


## Learning Curve
Learning curve comparison between different methods in an invisible setting.

<img src="https://i.imgur.com/l5UC3qa.png" width="600" />

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```bibtex
@inproceedings{chen2019crowd,
  title={Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning},
  author={Chen, Changan and Liu, Yuejiang and Kreiss, Sven and Alahi, Alexandre},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={6015--6022},
  year={2019},
  organization={IEEE}
}
```
