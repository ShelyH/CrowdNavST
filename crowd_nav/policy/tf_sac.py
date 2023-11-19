import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from crowd_nav.utils.utils_sac import ReplayBuffer
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.utils.comments import ATCBasicTfencoder, mlp, device
from crowd_nav.utils.cbp import CompactBilinearPooling


def reshapeT(T, seq_length):
    # print(T)
    shape = T.size()[0:]
    # print(shape)

    return T.reshape((seq_length, *shape))


class RNNBase(nn.Module):
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self):
        super(RNNBase, self).__init__()

        self.temporal_input_dim = 2
        self.gru_hidden_dim = 32
        self.gru = nn.GRU(self.temporal_input_dim, self.gru_hidden_dim * 2)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    # x: [seq_len, nenv, 6 or 30 or 36, ?]
    # hxs: [1, nenv, 6 or 30 or 36, ?]
    # masks: [1, nenv, 1]
    def _forward_gru(self, x, hxs):
        # for acting model, input shape[0] == hidden state shape[0]
        if x.size(0) == hxs.size(0):  # True
            # use env dimension as batch
            # [1, 1, agent_num, ?] -> [1, 1*agent_num, ?]
            seq_len, agent_num, _ = x.size()
            # print(x.size())
            x = x.view(seq_len, agent_num, -1)
            # print(hxs.size())
            # print('1', hxs)
            # hxs_times_masks = hxs * (masks.view(seq_len, nenv, 1, 1))
            hxs = hxs.view(seq_len, agent_num, -1)
            # we already unsqueezed the inputs in SRNN forward function
            # print(x.shape, hxs.shape)  # torch.Size([1, 4, 2]) torch.Size([1, 4, 320])
            x, hxs = self.gru(x, hxs)
            x = x.view(seq_len, agent_num, -1)
            hxs = hxs.view(seq_len, 1, agent_num, -1)
        # during update, input shape[0] * nsteps (30) = hidden state shape[0]
        else:
            # x.size():[30, 1, agent_num, ?] -> [30, 1*agent_num, ?]
            # hxs.size():[1, 1, agent_num, ?] -> [1, 1*agent_num, ?]
            # T: seq_len, N: nenv, agent_num: node num or edge num
            T, agent_num, _ = x.size()

            # seq_len, nenv, agent_num, _ = x.size()
            x = x.view(T, agent_num, -1)
            # print(hxs.shape)
            hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), -1)
            # print(hxs.shape)
            rnn_scores, hxs = self.gru(x, hxs)

            x = torch.cat([rnn_scores], dim=0)
            x = x.view(T, agent_num, -1)

            hxs = hxs.view(1, 1, agent_num, -1)

        return x, hxs


class DQFunc(RNNBase):
    def __init__(self, self_state_dim, joint_state_dim, in_mlp_dims, value_dims, action_dims, act_steps=1,
                 act_fixed=False):
        super(DQFunc, self).__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims  # [100, 2]

        self.spatial_input_dim = 2
        self.temporal_input_dim = 2
        self.act_steps = act_steps
        self.act_fixed = act_fixed
        # spatial TransformerEncoder
        self.spatial_in_mlp = mlp(self.spatial_input_dim, self.in_mlp_dims)
        # self.temporal_in_mlp = mlp(self.temporal_input_dim, self.in_mlp_dims)
        self.temporal_in_gru = nn.GRU(self.temporal_input_dim, self.in_mlp_dims[-1])
        # self.fuse_layer = CompactBilinearPooling(1, 1, self.in_mlp_dims[-1])
        # self.z = mlp(self.in_mlp_dims[-1] * 2, self.in_mlp_dims)
        self.spatialTemporalfused = mlp(self.in_mlp_dims[-1] * 2 + self.self_state_dim, self.in_mlp_dims)
        self.attTF = ATCBasicTfencoder(self.in_mlp_dims[-1], in_mlp_dims, epsilon=0.05,
                                       last_relu=True, act_steps=self.act_steps, act_fixed=self.act_fixed)
        self.att_encoder_layer = nn.TransformerEncoderLayer(d_model=self.in_mlp_dims[-1], nhead=2, dim_feedforward=150)
        self.att_encoder = nn.TransformerEncoder(self.att_encoder_layer, num_layers=3)

        self.ac_in = in_mlp_dims[-1] + self.self_state_dim  # 50 + 6
        # self.ac_in = self.ac_in * self.human_num

        self.q1 = mlp(self.ac_in + action_dims[-1], value_dims)  # 58,150,100,100,1
        self.q2 = mlp(self.ac_in + action_dims[-1], value_dims)  # 58,150,100,100,1

        self.attention_weights = None
        self.step_cnt = 0

    def get_step_cnt(self):
        return self.step_cnt
        pass

    def forward(self, state, action):
        robot_state = state[:, :, 0:7].clone().detach()
        self_state = state[:, 0:1, 0:7].clone().detach()
        spatial_edges = state[:, :, 7:9]
        temporal_edges = state[:, :, 9:11]
        # print(spatial_edges)
        size = spatial_edges.shape
        # print(size)  # torch.Size([4, 2])

        # spatial
        spatial_future = self.spatial_in_mlp(spatial_edges)
        # temporal
        # temporal_future = self.temporal_in_mlp(temporal_edges)
        h0 = torch.zeros(1, size[1], self.in_mlp_dims[-1])
        # print(temporal_edges.shape)
        temporal_future, hn = self.temporal_in_gru(temporal_edges, h0)
        # print(spatial_future.shape, temporal_future.shape)  # torch.Size([4, 50]) torch.Size([4, 50])
        fused_state = torch.cat([spatial_future, temporal_future, robot_state], dim=-1)
        # print(fused_state.shape)  # torch.Size([1, 4, 306])
        # fuse_state = self.fuse_layer(spatial_future.unsqueeze(1), temporal_future.unsqueeze(1))
        # print(fuse_state.shape)
        # z = torch.sigmoid(torch.cat([spatial_future, temporal_future], dim=-1))
        # print(z.shape)  # torch.Size([?, 5, 50])
        # z = self.z(z)

        # fuse_state = z * spatial_future + (1 - z) * temporal_future
        # fused_state = torch.cat([fuse_state, robot_state], dim=-1)
        fused_state_output = self.spatialTemporalfused(fused_state)
        # fused_state_output = fused_state_output.unsqueeze(0)
        # fused_state_output = fused_state_output.transpose(0, 1).contiguous()
        # print(fused_state_output.shape)  # torch.Size([4, 50])
        # act_h0 = torch.zeros([size[0] * size[1], self.in_mlp_dims[-1]]).cpu()  # 500 x 50
        att_in_mlp_output, _ = self.attTF(fused_state_output)
        att_in_mlp_output = att_in_mlp_output.view(size[0], -1, self.in_mlp_dims[-1])
        # att_in_mlp_output = att_in_mlp_output.unsqueeze(0)
        # att_in_mlp_output = att_in_mlp_output.transpose(0, 1).contiguous()
        # print(att_in_mlp_output.shape)
        att_in_mlp_output = att_in_mlp_output.transpose(0, 1).contiguous()
        tfencoder_output = self.att_encoder(att_in_mlp_output)
        tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()
        env_info = torch.mean(tfencoder_output, dim=1, keepdim=True)
        # print(env_info.shape, self_state.shape, action.shape)
        value_in = torch.cat([env_info, self_state, action], dim=-1)
        # print(value_in)
        q1, q2 = self.q1(value_in), self.q2(value_in)
        # value_in = torch.cat([joint_state, action], dim=-1)
        # print(value_in.shape, q1.shape)
        return q1, q2


class ST_Policy(RNNBase):
    def __init__(self, self_state_dim, joint_state_dim, in_mlp_dims, action_dims, act_steps=1, act_fixed=False):
        super(ST_Policy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.self_state_dim = self_state_dim
        self.global_state_dim = in_mlp_dims[-1]
        self.joint_state_dim = joint_state_dim
        self.in_mlp_dims = in_mlp_dims  # [100, 50]
        self.spatial_input_dim = 2
        self.temporal_input_dim = 2

        self.act_steps = act_steps
        self.act_fixed = act_fixed
        # spatial TransformerEncoder
        self.spatial_in_mlp = mlp(self.spatial_input_dim, self.in_mlp_dims)
        # self.temporal_in_mlp = mlp(self.temporal_input_dim, self.in_mlp_dims)
        self.temporal_in_gru = nn.GRU(self.temporal_input_dim, self.in_mlp_dims[-1])
        self.spatialTemporalfused = mlp(self.in_mlp_dims[-1] * 2 + self.self_state_dim, self.in_mlp_dims)
        # print(self.spatialTemporalfused)
        # self.fuse_layer = CompactBilinearPooling(1, 1, self.in_mlp_dims[-1])
        # self.z = mlp(self.in_mlp_dims[-1] * 2, self.in_mlp_dims)
        self.attTF = ATCBasicTfencoder(self.in_mlp_dims[-1], in_mlp_dims, epsilon=0.05,
                                       last_relu=True, act_steps=self.act_steps, act_fixed=self.act_fixed)
        self.att_encoder_layer = nn.TransformerEncoderLayer(d_model=self.in_mlp_dims[-1], nhead=2, dim_feedforward=150)
        self.att_encoder = nn.TransformerEncoder(self.att_encoder_layer, num_layers=3)

        self.action_input_dim = in_mlp_dims[-1] + self.self_state_dim  # 50 + 6

        self.action_mean = mlp(self.action_input_dim, action_dims)  # 56,150,100,100,1
        self.action_std = mlp(self.action_input_dim, action_dims)  # 56,150,100,100,1

        self.attention_weights = None
        self.step_cnt = 0

    def forward(self, state):
        # print(state.shape)
        # radius, dx, dy, vx, vy, dg, da, px1, py1, vx1, vy1, radius1
        robot_state = state[:, :, 0:7].clone().detach()
        self_state = state[:, 0:1, 0:7].clone().detach()
        spatial_edges = state[:, :, 7:9]
        temporal_edges = state[:, :, 9:11]
        # print(spatial_edges.shape)
        # print(state)
        # print(robot_state)
        # print(spatial_edges)
        size = spatial_edges.shape

        # print(size)  # torch.Size([4, 2])
        # spatial
        spatial_future = self.spatial_in_mlp(spatial_edges)
        # temporal
        # temporal_future = self.temporal_in_mlp(temporal_edges)
        h0 = torch.zeros(1, size[1], self.in_mlp_dims[-1])
        # print(h0.size())  # torch.Size([1, 5, 50])
        # print(temporal_edges.shape)  # torch.Size([？, 5, 2])
        temporal_future, hn = self.temporal_in_gru(temporal_edges, h0)
        # print(temporal_future.shape)
        # f = spatial_future * temporal_future
        # torch.Size([?, 5, 50]) torch.Size([？, 5, 7])
        # print(spatial_future.shape, temporal_future.shape, robot_state.shape)

        # fuse_state = self.fuse_layer(spatial_future.unsqueeze(1), temporal_future.unsqueeze(1))
        # print(fuse_state.shape)
        # z = torch.sigmoid(torch.cat([spatial_future, temporal_future], dim=-1))
        # print(z.shape)  # torch.Size([?, 5, 50])
        # z = self.z(z)

        # fuse_state = z * spatial_future + (1 - z) * temporal_future
        # fused_state = torch.cat([fuse_state, robot_state], dim=-1)
        fused_state = torch.cat([spatial_future, temporal_future, robot_state], dim=-1)
        fused_state_output = self.spatialTemporalfused(fused_state)
        # print(fused_state_output.shape)
        # fused_state_output = fused_state_output.transpose(0, 1).contiguous()

        att_in_mlp_output, _ = self.attTF(fused_state_output)

        att_in_mlp_output = att_in_mlp_output.view(size[0], -1, self.in_mlp_dims[-1])

        att_in_mlp_output = att_in_mlp_output.transpose(0, 1).contiguous()
        tfencoder_output = self.att_encoder(att_in_mlp_output)
        tfencoder_output = tfencoder_output.transpose(0, 1).contiguous()
        # print(tfencoder_output.shape)
        env_info = torch.mean(tfencoder_output, dim=1, keepdim=True)

        joint_state = torch.cat([env_info, self_state], dim=-1)
        # print(joint_state.shape)  # torch.Size([1, 1, 57])
        action_mean = self.action_mean(joint_state)
        action_logstd = torch.clamp(self.action_std(joint_state), min=-20, max=2)
        action_logstd = action_logstd.exp()

        return action_mean, action_logstd

    def sample(self, state, deterministic=False):
        # print(state)
        a_mean, standard_log = self.forward(state)
        # print(a_mean,standard_log)
        dist = Normal(a_mean, standard_log)
        if deterministic:
            position_x = dist.mean
        else:
            position_x = dist.rsample()
        A_ = torch.tanh(position_x.squeeze(0))
        log_prob = dist.log_prob(position_x) - torch.log(1 - A_.pow(2) + 1e-6)
        # print(A_.shape)
        return A_, log_prob.sum(-1, keepdim=True)


class TFSAC(MultiHumanRL):
    def __init__(self, action_dim, state_dim, capacity=None, batchsize=None,
                 lr=0.0005, gamma=0.99, tau=5e-3, update_interval=1, target_entropy=None):
        super(TFSAC, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.action_dims = [100, 50, 2]
        self.sort_mlp_dims = [100, 50]
        self.sort_mlp_attention = [50, 50, 1]
        self.value_dims = [100, 50, 1]
        self.in_mlp_dims = [100, 50]
        self.self_state_dim = 7
        self.human_state_dim = 5
        self.input_dim = self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.om_channel_size = 3

        self.actor = ST_Policy(self.self_state_dim, self.joint_state_dim, self.in_mlp_dims, self.action_dims)

        self.qfunsac = DQFunc(self.self_state_dim, self.joint_state_dim, self.in_mlp_dims, self.value_dims,
                              self.action_dims).to(device)

        self.target_q = copy.deepcopy(self.qfunsac)
        self.c_net_target = self.target_q
        self.critic = self.qfunsac
        ####################
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # print(lr)
        self.q_optimizer = torch.optim.Adam(self.qfunsac.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=0.005)

        self.replay_pool = ReplayBuffer(capacity=capacity)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # print(self.actor)

    def predict(self, state, deterministic=False):
        with torch.no_grad():
            # print(state)
            # state = state['spatial_edges']
            state = torch.FloatTensor(state).unsqueeze(0)
            # print(deterministic)
            action, logstd = self.actor.sample(state, deterministic)

        return action

    def soft_update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(),
                                               self.qfunsac.parameters()):
                target_q_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def optimize(self, update):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0
        # print(self.replay_pool.__len__())
        if self.replay_pool.__len__() > 2000:
            state, action, reward, next_state, done = self.replay_pool.sample(self.batchsize)
            state_batch = torch.FloatTensor(state)
            nextstate_batch = torch.FloatTensor(next_state)
            action_batch = torch.FloatTensor(action)
            reward_batch = torch.FloatTensor(reward).unsqueeze(1).unsqueeze(1)
            done_batch = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(1)
            # print(action_batch.shape, reward_batch.shape)
            # update q-funcs
            with torch.no_grad():
                # print(nextstate_batch.shape)
                nextaction_batch, logprobs_batch = self.actor.sample(nextstate_batch, deterministic=False)
                q_t1, q_t2 = self.target_q(nextstate_batch, nextaction_batch)
                # print(reward_batch.shape)
                # take min to mitigate positive bias in q-function training
                q_target = torch.min(q_t1, q_t2)
                value_target = reward_batch + (1 - done_batch) * self.gamma * (
                        q_target - self.alpha * logprobs_batch)

            # JQ = 𝔼(st,at)~D[1/2(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            q_1, q_2 = self.qfunsac(state_batch, action_batch)
            # print(q_1.shape, q_2.shape)
            # print(value_target.shape)
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
            self.actor.load_state_dict(torch.load(actor_net_path))
