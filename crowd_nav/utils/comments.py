#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CrowdNavRL 
@File    ：comments.py
@Author  ：HHD
@Date    ：2023/1/1 下午9:32 
"""
import torch
import torch.nn as nn

from crowd_nav.utils.transformer import TransformerEncoder

cuda = False
device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            # layers.append(nn.BatchNorm1d(mlp_dims[i + 1]))
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ATCBasicTfencoder(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dims, n_head, max_ponder=3, epsilon=0.05, act_steps=3,
                 act_fixed=False):
        super(ATCBasicTfencoder, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rnn_hidden_dim = rnn_hidden_dims[-1]
        self.epsilon = epsilon
        self.transformer = TransformerEncoder(source_dims=input_dim, n_heads=n_head, k_dims=64, v_dims=64, layer_cnt=1)
        self.transition_layer = nn.Linear(input_dim, rnn_hidden_dims[-1])

        self.max_ponder = max_ponder
        self.ponder_linear = nn.Linear(rnn_hidden_dims[-1], 1)

        self.act_fixed = act_fixed
        self.act_steps = act_steps

    def forward(self, input):
        # Pre-allocate variables
        # time_size, batch_size, input_dim_size = input.size()
        size = input.shape
        input_ = input.view(-1, size[2])  # torch.Size([4, 2])
        accum_p = 0
        accum_hx = torch.zeros([input_.shape[0], self.rnn_hidden_dim]).to(device)
        step_count = 0
        for act_step in range(self.max_ponder):
            hx = self.transformer(input)
            hx = self.transition_layer(hx)
            hx = hx.view(-1, self.rnn_hidden_dim)
            halten_p = torch.sigmoid(self.ponder_linear(hx))  # halten state
            accum_p += halten_p
            accum_hx += halten_p * hx
            step_count += 1
            selector = (accum_p < 1 - self.epsilon).data
            if self.act_fixed:
                if step_count >= self.act_steps:
                    break
            else:
                if not selector.any():  # selector has no true elements
                    break

        hx = accum_hx / step_count

        return hx, step_count
