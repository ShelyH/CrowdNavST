# CrowdNavRL


This repository contains the code for the following papers:

- [Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning](https://arxiv.org/abs/1609.07845).
- [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1805.01956).
- [Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).
- [Adaptive Environment Modeling Based Reinforcement Learning for Collision Avoidance in Complex Scenes](https://arxiv.org/abs/2203.07709).
- [Relational Graph Learning for Crowd Navigation, IROS, 2020](https://github.com/ChanganVR/RelationalGraphLearning).
- [Social NCE: Contrastive Learning of Socially-aware Motion Representations, ICCV, 2021](https://github.com/vita-epfl/social-nce).


## Abstract

Ensuring robots can move safely and adhere to social norms in dynamic human environments is a crucial step towards robot
autonomous decision-making. In existing work, double serial separate modules are generally used to capture spatial and
temporal interactions, respectively. However, such methods lead to extra difficulties in improving the utilization of
spatio-temporal features and reducing the conservatism of navigation policy. In light of this, this paper proposes a
spatiotemporal transformer-based policy optimization algorithm to more effectively preserve the human-robot
interactions. Specifically, a gated embedding mechanism is introduced to effectively fuses the spatial and temporal
representations by integrating both modalities at the feature level. Then Transformer is leveraged to encode the
spatio-temporal semantic information, with the hope of finding the optimal navigation policy. Finally, a combination of
spatio-temporal Transformer and self-adjusting policy entropy significantly reduce the conservatism of navigation
policies. Experimental results demonstrate the priority of the proposed algorithm over the state-of-the-art methods.

