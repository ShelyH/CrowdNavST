import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.utils.utils_sac import ReplayPool

# from crowd_nav.utils.utils_sac import ReplayPool

gpu_is_True = False
device = torch.device("cuda" if torch.cuda.is_available() and gpu_is_True else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
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


class PolicyNetGaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PolicyNetGaussian, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)

        self.layer_4_mean = nn.Linear(hidden_size, 2)
        self.layer_4_standard_log = nn.Linear(hidden_size, action_dim)

    def forward(self, s):
        hidden_layer_1 = F.relu(self.layer1(s))
        hidden_layer_2 = F.relu(self.layer2(hidden_layer_1))
        hidden_layer_3 = F.relu(self.layer3(hidden_layer_2))

        return self.layer_4_mean(hidden_layer_3), \
            torch.clamp(self.layer_4_standard_log(hidden_layer_3), min=-20, max=2)

    def sample(self, s, deterministic):
        a_mean, standard_log = self.forward(s)
        a_std = standard_log.exp()
        dist = Normal(a_mean, a_std)
        if deterministic:
            position_x = dist.mean
        else:
            position_x = dist.rsample()

        A_ = torch.tanh(position_x)
        log_prob = dist.log_prob(position_x) - torch.log(1 - A_.pow(2) + 1e-6)

        return A_, log_prob.sum(1, keepdim=True), torch.tanh(a_mean)


class DQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DQFunc, self).__init__()
        # print(state_dim, action_dim)  # 19, 2
        self.network1 = MLP(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLP(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        # print(state.shape, action.shape) # [30,19],[30,2]
        x = torch.cat((state, action), dim=1)
        # print(x.shape)
        return self.network1(x), self.network2(x)


class USAC(MultiHumanRL):
    def __init__(self, action_dim, state_dim, capacity, batchsize, lr=5e-4, gamma=0.99,
                 tau=0.005, hidden_size=256, update_interval=1, target_entropy=None):
        # self.multiagent_training = None
        super(MultiHumanRL, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval
        # print(self.target_entropy)
        # torch.manual_seed(seed)
        # aka critic
        ####################
        self.qfunsac = DQFunc(state_dim, action_dim, hidden_size).to(device)
        self.target_q = copy.deepcopy(self.qfunsac)
        self.critic = self.qfunsac
        self.c_net_target = self.target_q
        ####################
        self.target_q.eval()
        for p in self.target_q.parameters():
            p.requires_grad = False

        # print(state_dim)
        # aka actor
        self.policynet = PolicyNetGaussian(state_dim, action_dim, hidden_size)
        self.actor = self.policynet
        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.q_optimizer = torch.optim.Adam(self.qfunsac.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=0.005)
        # print(state_dim)
        self.replay_pool = ReplayPool(action_dim=action_dim, state_dim=state_dim, capacity=capacity)

    def configure(self, config):
        self.set_common_parameters(config)
        self.multiagent_training = config.getboolean('sac_rl', 'multiagent_training')

    def set_common_parameters(self, config):
        self.kinematics = config.get('action_space', 'kinematics')

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q.parameters(), self.qfunsac.parameters()):
                target_q_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def predict(self, s, deterministic):
        # print(s.human_states)
        self.last_state = s
        robot_fullstate = np.array(s.self_state.out_state())
        human_state = []
        # print(len(s.human_states))
        for i in range(len(s.human_states)):
            human_state.append(np.array([s.human_states[i].px,
                                         s.human_states[i].px, s.human_states[i].vx,
                                         s.human_states[i].vy, s.human_states[i].radius]))
        s = np.ravel(human_state)
        s = np.concatenate((s, robot_fullstate), axis=0)
        s = torch.tensor(s, dtype=torch.float32)
        with torch.no_grad():
            state = torch.Tensor(np.array(s)).view(1, -1).to(device)
            action, _, _ = self.policynet.sample(state, deterministic)

        return action

    def optim_sac(self):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0
        # print(self.replay_pool.__len__())
        samples = self.replay_pool.sample(self.batchsize)
        # print(device)
        state_batch = torch.FloatTensor(np.array(samples.state)).to(device)
        nextstate_batch = torch.FloatTensor(np.array(samples.nextstate)).to(device)
        action_batch = torch.FloatTensor(np.array(samples.action)).to(device)
        reward_batch = torch.FloatTensor(np.array(samples.reward)).to(device).unsqueeze(1)
        done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)
        # update q-funcs
        # print(masks_batch)
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policynet.sample(nextstate_batch, deterministic=False)

            q_t1, q_t2 = self.target_q(nextstate_batch, nextaction_batch)

            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            # print(done_batch.shape,m.shape)
            value_target = reward_batch + (1 - done_batch) * self.gamma * (
                    q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.qfunsac(state_batch, action_batch)
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

        # q_b1, q_b2=self.actor_critic.eval
        q_b1, q_b2 = self.qfunsac(state_batch, action_batch)

        qval_batch = torch.min(q_b1, q_b2)
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
        self.update_target()

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
