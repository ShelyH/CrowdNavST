import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from crowd_nav.policy.transformer import Transformer
from crowd_nav.utils.utils_sac import ReplayBuffer
from crowd_nav.policy.multi_human_rl import MultiHumanRL


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


def reshapeT(T, seq_length):
    shape = T.size()[0:]

    return T.reshape((seq_length, *shape))


class FeatureExtractNet(nn.Module):
    def __init__(self, in_mlp_dims, device):
        super().__init__()
        self.robot_state_dim = 7
        self.spatial_input_dim = 2
        self.temporal_input_dim = 2

        self.device = device
        self.spatial_in_mlp = mlp(self.spatial_input_dim, in_mlp_dims).to(device)
        self.temporal_in_gru = nn.GRU(self.temporal_input_dim, in_mlp_dims[-1]).to(device)
        self.spatialTemporalfused = mlp(in_mlp_dims[-1], in_mlp_dims).to(device)
        self.fc = mlp(in_mlp_dims[-1] + self.robot_state_dim, in_mlp_dims).to(device)
        self.transformer = Transformer(d_model=in_mlp_dims[-1], seq_length=None, depth=3, heads=5, dim_head=256,
                                       mlp_dim=in_mlp_dims[-1]).to(device)
        # self.transformer = transformer.TransformerEncoder(self.in_mlp_dims[-1], 5, 2)

    def forward(self, state, hidden_state):
        # radius, vx, vy, dg, rot_g, r_rot, radius_sum, hr_rot, da, px1, py1, vx1, vy1, radius1
        robot_state_node = state[:, :, 0:7].clone().detach().to(self.device)
        spatial_edges = state[:, :, 9:11].to(self.device)
        temporal_edges = state[:, :, 11:13].to(self.device)
        self.temporal_in_gru.flatten_parameters()
        # spatial  # torch.Size([?, 50])
        spatial_future = self.spatial_in_mlp(spatial_edges)
        # temporal  # torch.Size([?, 50])
        temporal_future, hn = self.temporal_in_gru(temporal_edges, hidden_state)
        # ST-FUSE
        z = torch.sigmoid(spatial_future + temporal_future)
        H = z * spatial_future + (1 - z) * temporal_future
        fused_state = self.spatialTemporalfused(H)
        # print(fused_state.shape)
        # fused_state = torch.cat([spatial_future, temporal_future, robot_state_node], dim=-1)
        fused_state = torch.cat([fused_state, robot_state_node], dim=-1)
        # print(fused_state.shape)
        fused_state = self.fc(fused_state)
        tfencoder_output = self.transformer(fused_state.clone().detach())

        return tfencoder_output, hn


class ST_Policy(nn.Module):
    def __init__(self, self_state_dim, in_mlp_dims, action_dims, device):
        super(ST_Policy, self).__init__()
        self.device = device
        self.self_state_dim = self_state_dim
        self.st_transformer = FeatureExtractNet(in_mlp_dims, device)
        self.action_input_dim = in_mlp_dims[-1] + self.self_state_dim  # 50 + 6
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.action_mean = mlp(self.action_input_dim, action_dims)  # 56,150,100,100,1
        self.action_std = mlp(self.action_input_dim, action_dims)  # 56,150,100,100,1

        self.attention_weights = None
        self.step_cnt = 0

    def forward(self, state, h):
        # radius, dx, dy, vx, vy, dg, da, px1, py1, vx1, vy1, radius1
        self_state = state[:, :, 0:7].clone().detach().to(self.device)
        h = h.to(self.device)
        att_in_mlp_output, hn = self.st_transformer(state, h)
        # env_info = torch.mean(att_in_mlp_output, dim=1, keepdim=True)
        joint_state = torch.cat([att_in_mlp_output, self_state], dim=-1)
        joint_state = joint_state.permute([0, 2, 1])
        env_info = self.avgpool(joint_state)
        env_info = env_info.permute([0, 2, 1])
        # print(env_info.shape)
        # joint_state = torch.cat([env_info], dim=-1)
        action_mean = self.action_mean(env_info)
        action_logstd = torch.clamp(self.action_std(env_info), min=-20, max=2)
        action_logstd = action_logstd.exp()

        return action_mean, action_logstd, hn

    def sample(self, state, h, deterministic=False):
        a_mean, standard_log, hn = self.forward(state, h)
        dist = Normal(a_mean, standard_log)
        if deterministic:
            position_x = dist.mean
        else:
            position_x = dist.rsample()
        A_ = torch.tanh(position_x.squeeze(0))
        log_prob = dist.log_prob(position_x) - torch.log(1 - A_.pow(2) + 1e-6).to(self.device)

        return A_, log_prob.sum(-1, keepdim=True), hn


class QFunc(nn.Module):
    def __init__(self, self_state_dim, in_mlp_dims, value_dims, action_dims, device):
        super(QFunc, self).__init__()
        self.device = device
        self.st_transformer = FeatureExtractNet(in_mlp_dims, device)
        self.q1 = mlp(in_mlp_dims[-1] + self_state_dim + action_dims[-1], value_dims)  # 58,150,100,100,1
        self.q2 = mlp(in_mlp_dims[-1] + self_state_dim + action_dims[-1], value_dims)  # 58,150,100,100,1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.attention_weights = None
        self.step_cnt = 0

    def forward(self, state_batch, action_batch, h_batch):
        self_state_batch = state_batch[:, :, 0:7].clone().detach().to(self.device)
        tfencoder_output, _ = self.st_transformer(state_batch, h_batch)
        joint_state = torch.cat([tfencoder_output, self_state_batch], dim=-1)
        # print(joint_state.shape)
        joint_state = joint_state.permute([0, 2, 1])
        env_info = self.avgpool(joint_state)
        env_info = env_info.permute([0, 2, 1])
        # print(env_info.shape)
        value_in = torch.cat([env_info, action_batch], dim=-1)
        # print(value_in.shape)
        q1, q2 = self.q1(value_in), self.q2(value_in)

        return q1, q2


class TFRNNSAC(MultiHumanRL):
    def __init__(self, action_dim, capacity=None, batchsize=None, in_mlp_dims=None, action_dims=None,
                 value_dims=None, device=None, target_entropy=None, lr=0.0005, gamma=0.99, tau=5e-3, update_interval=1):
        super(TFRNNSAC, self).__init__()
        self.name = 'tf_rnn_sac'
        self.gamma = gamma
        self.tau = tau
        # self.state_dim = state_dim
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.in_mlp_dims = in_mlp_dims
        self.action_dims = action_dims
        self.value_dims = value_dims
        self.self_state_dim = 7
        self.human_state_dim = 7
        self.device = device
        self.input_dim = self.joint_state_dim = self.self_state_dim + self.human_state_dim

        self.actor = ST_Policy(self.self_state_dim, self.in_mlp_dims, self.action_dims, self.device).to(device)
        self.qfunsac = QFunc(self.self_state_dim, self.in_mlp_dims, self.value_dims, self.action_dims, self.device).to(
            device)

        self.target_q = copy.deepcopy(self.qfunsac)
        self.c_net_target = self.target_q
        self.critic = self.qfunsac
        ####################
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.q_optimizer = torch.optim.Adam(self.qfunsac.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=0.005)
        self.replay_pool = ReplayBuffer(capacity=capacity)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    # def set_device(self, device):
    #     self.device = device
    #     self.actor.to(device)
    #     self.qfunsac.to(device)

    def get_action(self, state, h, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, logstd, hn = self.actor.sample(state, h, deterministic)

        return action, hn

    def soft_update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(),
                                               self.qfunsac.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def optimize(self, update, h_batch):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0
        if self.replay_pool.__len__() > 5000:
            state, action, reward, next_state, done = self.replay_pool.sample(self.batchsize)
            state_batch = torch.FloatTensor(state).to(self.device)
            nextstate_batch = torch.FloatTensor(next_state).to(self.device)
            action_batch = torch.FloatTensor(action).to(self.device)
            reward_batch = torch.FloatTensor(reward).unsqueeze(1).unsqueeze(1).to(self.device)
            done_batch = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(1).to(self.device)
            # update q-funcs
            with torch.no_grad():
                nextaction_batch, logprobs_batch, _ = self.actor.sample(nextstate_batch, h_batch, deterministic=False)
                q_t1, q_t2 = self.target_q(nextstate_batch, nextaction_batch, h_batch)

                # take min to mitigate positive bias in q-function training
                q_target = torch.min(q_t1, q_t2).to(self.device)

                value_target = reward_batch + (1 - done_batch) * self.gamma * (
                        q_target - self.alpha * logprobs_batch)

            # JQ = 𝔼(st,at)~D[1/2(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            q_1, q_2 = self.qfunsac(state_batch, action_batch, h_batch)

            q1_loss_step = F.mse_loss(q_1, value_target)
            q2_loss_step = F.mse_loss(q_2, value_target)
            q_loss_step = q1_loss_step + q2_loss_step

            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.qfunsac.parameters():
                p.requires_grad = False

            action_batch, logprobs_batch, _ = self.actor.sample(state_batch, h_batch, deterministic=False)

            q_b1, q_b2 = self.qfunsac(state_batch, action_batch, h_batch)
            qval_batch = torch.min(q_b1, q_b2)
            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            pi_loss_step = (self.alpha * logprobs_batch - qval_batch).mean()
            alpha_loss_step = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()

            self.optimizer.zero_grad()
            pi_loss_step.backward()
            self.optimizer.step()

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
            torch.save(self.actor.state_dict(), actor_net_path)
            torch.save(self.critic.state_dict(), critic_net_path)
        elif op == "load":
            # self.critic.load_state_dict(torch.load(critic_net_path))
            # self.c_net_target.load_state_dict(torch.load(critic_net_path))
            self.actor.load_state_dict(torch.load(actor_net_path, map_location='cpu'))
