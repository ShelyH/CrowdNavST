import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import softmax
from torch.distributions import Normal
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.utils.utils_sac import ReplayBuffer

gpu_is_True = False
device = torch.device("cuda" if torch.cuda.is_available() and gpu_is_True else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, self_state_dim):
        super(DQFunc, self).__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = 256
        self.mlp_output_dim = 8
        self.mlp1 = MLP(input_dim=state_dim, output_dim=256)
        self.mlp2 = MLP(input_dim=256, output_dim=self.mlp_output_dim)
        self.attention = MLP(input_dim=256 * 2, output_dim=1)
        self.joint_state_dim = self.mlp_output_dim + self.self_state_dim
        self.linear1 = nn.Linear(action_dim + self.joint_state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.Q1 = nn.Linear(hidden_size, 1)
        self.Q2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
            contiguous().view(-1, self.global_state_dim)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)

        att = self.attention(attention_input)

        scores = att.view(size[0], size[1], 1).squeeze(dim=2)
        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()

        weights = softmax(scores, dim=1).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)

        x = torch.cat([joint_state, action], -1)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x1 = self.Q1(x)
        x2 = self.Q2(x)

        return x1, x2


class PolicyNetGaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, self_state_dim):
        super(PolicyNetGaussian, self).__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = 256
        self.mlp_output_dim = 8
        self.mlp1 = MLP(input_dim=state_dim - self_state_dim, output_dim=256)
        self.mlp2 = MLP(input_dim=256, output_dim=self.mlp_output_dim)
        self.attention = MLP(input_dim=256 * 2, output_dim=1)
        self.joint_state_dim = self.mlp_output_dim + self.self_state_dim
        self.linear1 = nn.Linear(self.joint_state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        self_state = state[:, 0, :self.self_state_dim]
        human_state = state[:, :, self.self_state_dim:]
        size = human_state.shape
        mlp1_output = self.mlp1(human_state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)

        global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
            contiguous().view(-1, self.global_state_dim)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)

        att = self.attention(attention_input)

        scores = att.view(size[0], size[1], 1).squeeze(dim=2)
        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        # scores_exp = torch.exp(scores) * (scores != 0).float()

        weights = softmax(scores, dim=1).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        x = torch.relu(self.linear1(joint_state))
        x = torch.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state, deterministic):
        a_mean, standard_log = self.forward(state)
        # print(a_mean)
        a_std = standard_log.exp()
        dist = Normal(a_mean, a_std)
        if deterministic:
            position_x = dist.mean
        else:
            position_x = dist.rsample()
        A_ = torch.tanh(position_x)
        log_prob = dist.log_prob(position_x) - torch.log(1 - A_.pow(2) + 1e-6)
        return A_, log_prob.sum(1, keepdim=True), torch.tanh(a_mean)


class SARLSAC(MultiHumanRL):
    def __init__(self, action_dim, state_dim, capacity, batchsize, lr=5e-4, gamma=0.99, reward_scale=100,
                 tau=0.005, hidden_size=256, update_interval=1, self_state_dim=7, target_entropy=None):
        super(MultiHumanRL, self).__init__()
        self.name = 'SARL_SAC'
        self.gamma = gamma
        self.tau = tau
        self.self_state_dim = 7
        self.human_state_dim = 7
        self.reward_scale = reward_scale
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.target_entropy = target_entropy if target_entropy else -action_dim
        # aka critic
        self.qfunsac = DQFunc(self.self_state_dim + self.human_state_dim, action_dim,
                              hidden_size, self_state_dim).to(device)
        self.target_q = copy.deepcopy(self.qfunsac)
        self.critic = self.qfunsac
        self.c_net_target = self.target_q
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False
        # aka actor
        self.policynet = PolicyNetGaussian(self.self_state_dim + self.human_state_dim,
                                           action_dim, hidden_size, self_state_dim)
        self.actor = self.policynet
        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.q_optimizer = torch.optim.Adam(self.qfunsac.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=0.005)

        self.replay_buffer = ReplayBuffer(capacity)
        self.hard_update()

    def soft_update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(), self.qfunsac.parameters()):
                target_q_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def hard_update(self):
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(), self.qfunsac.parameters()):
                target_q_param.data.copy_(q_param.data)

    def predict(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)

            action, logstd, mean = self.actor.sample(state, deterministic)

        return action

    def optim_sac(self, update):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batchsize)
        state_batch = torch.FloatTensor(state)
        nextstate_batch = torch.FloatTensor(next_state)
        action_batch = torch.FloatTensor(action)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1)
        done_batch = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        # update q-funcs
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policynet.sample(nextstate_batch, deterministic=False)
            q_t1, q_t2 = self.target_q(nextstate_batch, nextaction_batch)

            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = self.reward_scale * reward_batch + (1 - done_batch) * self.gamma * (
                    q_target - self.alpha * logprobs_batch)

        # JQ = 𝔼(st,at)~D[1/2(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_1, q_2 = self.qfunsac(state_batch, action_batch.squeeze(1))

        q1_loss_step = F.mse_loss(q_1, value_target)
        q2_loss_step = F.mse_loss(q_2, value_target)
        q_loss_step = q1_loss_step + q2_loss_step

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        # update policy and temperature parameter
        for p in self.qfunsac.parameters():
            p.requires_grad = False
        action_batch, logprobs_batch, _ = self.policynet.sample(state_batch, deterministic=False)

        q_b1, q_b2 = self.qfunsac(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        pi_loss_step = (self.alpha * logprobs_batch - qval_batch).mean()
        alpha_loss_step = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()

        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        self.temp_optimizer.zero_grad()
        alpha_loss_step.backward()
        self.temp_optimizer.step()

        for p in self.qfunsac.parameters():
            p.requires_grad = True

        q1_loss += q1_loss_step.detach().item()
        q2_loss += q2_loss_step.detach().item()
        pi_loss += pi_loss_step.detach().item()
        alpha_loss += alpha_loss_step.detach().item()
        if update % self.update_interval == 0:
            self.soft_update_target()

        return q1_loss, q2_loss, pi_loss, alpha_loss

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save_load_model(self, op, path):
        actor_net_path = path + "/A_SAC.pt"
        critic_net_path = path + "/C_SAC.pt"

        if op == "save":
            torch.save(self.critic.state_dict(), critic_net_path)
            torch.save(self.actor.state_dict(), actor_net_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(critic_net_path))
            self.c_net_target.load_state_dict(torch.load(critic_net_path))
            self.actor.load_state_dict(torch.load(actor_net_path))

