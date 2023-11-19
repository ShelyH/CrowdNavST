import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from crowd_nav.policy.crossTransformer import TTransformer, self_Attention
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
    def __init__(self, self_state_dim, human_state_dim, joint_state_dim, in_mlp_dims, device):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims  # [100, 50]
        self.device = device
        self.time_frame = 5
        self.cross_Temstate_fc = mlp(self.human_state_dim, self.in_mlp_dims).to(device)
        self.crossTem = TTransformer(embed_size=self.in_mlp_dims[-1], heads=5, time_num=self.time_frame,
                                     device=self.device).to(device)
        self.embed_liner = nn.Linear(self.human_state_dim, self.in_mlp_dims[-1]).to(device)
        self.self_att = self_Attention(dim=self.in_mlp_dims[-1], heads=5, dim_head=256).to(device)
        self.ts = TTransformer(embed_size=self.in_mlp_dims[-1], heads=5, time_num=self.time_frame,
                               device=self.device).to(device)

    def forward(self, state):
        # radius, dx, dy, vx, vy, dg, radius_sum, da, px, py1, vx1, vy1, radius1
        # da, px1, py1, vx1, vy1, radius1
        # his_human_state = [b, time_seq, human_num, w]
        B, T, N, W = state[:, :, 7:].view(-1, self.time_frame, state.shape[1] // self.time_frame, 6).shape
        his_human_state = state[:, :, 7:].view(B, T, N, W).to(self.device)

        his_state = his_human_state[:, 0:4, :, :]
        # cross_Temstate = self.cross_Temstate_fc(his_state)
        his_state = self.cross_Temstate_fc(his_state)
        cross_Temstate_score = self.crossTem(his_state, his_state, his_state, pe=False)
        # spatial state
        # sp = [b, time_seq, human_num w]
        cur_pos_state = his_human_state[:, -1, :, :].view(-1, N, W).to(self.device)
        cur_pos_state = self.embed_liner(cur_pos_state)
        sp_state = self.self_att(cur_pos_state).unsqueeze(1)

        tfencoder_output = torch.cat([cross_Temstate_score, sp_state], dim=1).permute([0, 2, 1, 3])
        tfencoder_output = self.ts(tfencoder_output, tfencoder_output, tfencoder_output)

        return tfencoder_output


class ST_Policy(nn.Module):
    def __init__(self, self_state_dim, human_state_dim, joint_state_dim, in_mlp_dims, action_dims, device):
        super(ST_Policy, self).__init__()
        self.device = device
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.st_transformer = FeatureExtractNet(self_state_dim, human_state_dim, joint_state_dim, in_mlp_dims, device)
        self.action_input_dim = in_mlp_dims[-1]  # + self.self_state_dim  # 50 + 6
        self.fc = mlp(7, in_mlp_dims).to(device)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.action_mean = mlp(self.action_input_dim + self.self_state_dim, action_dims)  # 56,150,100,100,1
        self.action_std = mlp(self.action_input_dim + self.self_state_dim, action_dims)  # 56,150,100,100,1

        self.attention_weights = None
        self.step_cnt = 0

    def forward(self, state):
        # radius, dx, dy, vx, vy, dg, da, px1, py1, vx1, vy1, radius1
        self_state = state[:, :, 0:7].clone().detach().to(self.device)[:, -1:]
        att_in_mlp_output = self.st_transformer(state)
        env_info = torch.mean(att_in_mlp_output, dim=1)
        env_info = self.avgpool(env_info.permute([0, 2, 1]))

        joint_state = torch.cat([env_info.permute([0, 2, 1]), self_state], dim=-1)
        action_mean = self.action_mean(joint_state)
        action_logstd = torch.clamp(self.action_std(joint_state), min=-20, max=2)
        action_logstd = action_logstd.exp()

        return action_mean, action_logstd

    def sample(self, state, deterministic=False):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample()
        if deterministic:
            action = torch.tanh(mu)
        else:
            action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + 1e-6)

        return action.squeeze(0), log_prob.sum(-1, keepdim=True)


class QFunc(nn.Module):
    def __init__(self, self_state_dim, human_state_dim, joint_state_dim, in_mlp_dims, value_dims, action_dims, device):
        super(QFunc, self).__init__()
        self.self_state_dim = self_state_dim
        self.st_transformer = FeatureExtractNet(self_state_dim, human_state_dim, joint_state_dim, in_mlp_dims, device)
        self.fc = mlp(7, in_mlp_dims).to(device)
        self.ac_in = in_mlp_dims[-1]  # 50 + 6
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.q1 = mlp(self.ac_in + self.self_state_dim + action_dims[-1], value_dims)  # 58,150,100,100,1
        self.q2 = mlp(self.ac_in + self.self_state_dim + action_dims[-1], value_dims)  # 58,150,100,100,1

        self.attention_weights = None
        self.step_cnt = 0

    def forward(self, state_batch, action_batch):
        self_state = state_batch[:, :, 0:7].clone().detach().to(self.device)[:, -1:]
        att_in_mlp_output = self.st_transformer(state_batch)
        env_info = torch.mean(att_in_mlp_output, dim=1)
        env_info = self.avgpool(env_info.permute([0, 2, 1]))
        value_in = torch.cat([env_info.permute([0, 2, 1]), self_state, action_batch], dim=-1)
        # print(value_in.shape)
        q1, q2 = self.q1(value_in), self.q2(value_in)

        return q1, q2


class STRL(MultiHumanRL):
    def __init__(self, action_dim, capacity=None, batchsize=None, in_mlp_dims=None, action_dims=None,
                 value_dims=None, device=None, target_entropy=None, lr=0.0005, gamma=0.99, tau=5e-3, update_interval=1):
        super(STRL, self).__init__()
        self.name = 'STRL'
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
        self.human_state_dim = 6
        self.device = device
        self.input_dim = self.joint_state_dim = self.self_state_dim + self.human_state_dim

        self.actor = ST_Policy(self.self_state_dim, self.human_state_dim, self.joint_state_dim, self.in_mlp_dims,
                               self.action_dims, device).to(device)
        self.qfunsac = QFunc(self.self_state_dim, self.human_state_dim, self.joint_state_dim, self.in_mlp_dims,
                             self.value_dims, self.action_dims, self.device).to(device)

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

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, logstd = self.actor.sample(state, deterministic)

        return action

    def soft_update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(),
                                               self.qfunsac.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def optimize(self, update):
        """
        Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0
        if self.replay_pool.__len__() > 400:
            state, action, reward, next_state, done = self.replay_pool.sample(self.batchsize)

            state_batch = torch.FloatTensor(state).to(self.device)
            nextstate_batch = torch.FloatTensor(next_state).to(self.device)
            action_batch = torch.FloatTensor(action).to(self.device)
            reward_batch = torch.FloatTensor(reward).unsqueeze(1).unsqueeze(1).to(self.device)
            done_batch = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(1).to(self.device)
            # update q-funcs
            with torch.no_grad():
                nextaction_batch, logprobs_batch = self.actor.sample(nextstate_batch, deterministic=False)
                q_t1, q_t2 = self.target_q(nextstate_batch, nextaction_batch)

                # take min to mitigate positive bias in q-function training
                q_target = torch.min(q_t1, q_t2).to(self.device)

                value_target = reward_batch + (1 - done_batch) * self.gamma * (
                        q_target - self.alpha * logprobs_batch)
            # print(logprobs_batch.shape)
            # JQ = 𝔼(st,at)~D[1/2(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            q_1, q_2 = self.qfunsac(state_batch, action_batch)
            # print(q_1.shape, value_target.shape)
            q1_loss_step = F.mse_loss(q_1, value_target)
            q2_loss_step = F.mse_loss(q_2, value_target)
            q_loss_step = q1_loss_step + q2_loss_step

            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.qfunsac.parameters():
                p.requires_grad = False

            action_batch, logprobs_batch = self.actor.sample(state_batch, deterministic=False)

            q_b1, q_b2 = self.qfunsac(state_batch, action_batch)
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
