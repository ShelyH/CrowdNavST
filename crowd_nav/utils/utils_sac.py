import math
import os
import random
import torch
import numpy as np
from math import pi, e, sin, cos, tan, sqrt, atan2
from numpy.random import default_rng
from collections import namedtuple
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus
from crowd_sim.envs.utils.state import ObservableState, FullState

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done'))


def transform_rh(s):
    robot_fullstate = np.array(s[-1].out_state())
    human_state = []
    for i in range(len(s) - 1):
        human_state.append(np.array([s[i].px,
                                     s[i].px, s[i].vx,
                                     s[i].vy, s[i].radius]))
    s = np.ravel(human_state)
    s = np.concatenate((s, robot_fullstate), axis=0)
    s = torch.tensor(s, dtype=torch.float32)
    return s


def transform_h(s):
    human_state = []
    for i in range(len(s)):
        human_state.append(np.array([s[i].px,
                                     s[i].px, s[i].vx,
                                     s[i].vy, s[i].radius]))
    s = np.ravel(human_state)
    s = torch.tensor(s, dtype=torch.float32)
    return s


def fullstateTotensor(fullstate):
    fullstate = [float(x) for x in [fullstate.px, fullstate.py, fullstate.vx,
                                    fullstate.vy, fullstate.radius, fullstate.gx,
                                    fullstate.gy, fullstate.v_pref, fullstate.theta]]
    fullstate = torch.tensor(fullstate)

    return fullstate


def transform(state):
    if isinstance(state, ObservableState):
        state = ObservableState(state.px, state.py, state.vx, state.vy, state.radius)
    elif isinstance(state, FullState):
        state = FullState(state.px, state.py, state.vx, state.vy, state.radius,
                          state.gx, state.gy, state.v_pref, state.theta)
    else:
        raise ValueError('Type error')

    return state


def rotate_s(state):
    trans_self_state = transform(state.self_state)
    trans_human_states = [transform(human_state) for human_state in state.human_states]

    batch_states = torch.cat([torch.Tensor([trans_self_state + human_state])
                              for human_state in trans_human_states], dim=0)

    # FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    # --> radius, dx, dy, vx, vy, dg, da, px1, py1, vx1, vy1, radius1
    batch = batch_states.shape[0]

    px = batch_states[:, 0].reshape((batch, -1))
    py = batch_states[:, 1].reshape((batch, -1))
    rvx = batch_states[:, 2].reshape((batch, -1))
    rvy = batch_states[:, 3].reshape((batch, -1))

    px_h = batch_states[:, 9].reshape((batch, -1))
    py_h = batch_states[:, 10].reshape((batch, -1))
    vx_h = batch_states[:, 11].reshape((batch, -1))
    vy_h = batch_states[:, 12].reshape((batch, -1))
    dx = (batch_states[:, 5] - batch_states[:, 0]).reshape((batch, -1))

    # rotate
    dy = (batch_states[:, 6] - batch_states[:, 1]).reshape((batch, -1))
    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    rot = torch.atan2(batch_states[:, 6] - batch_states[:, 1], batch_states[:, 5] - batch_states[:, 0])
    rot_g = rot.view(-1, 1)
    v_pref = batch_states[:, 7].reshape((batch, -1))
    vx = (batch_states[:, 2] * torch.cos(rot) + batch_states[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (batch_states[:, 3] * torch.cos(rot) - batch_states[:, 2] * torch.sin(rot)).reshape((batch, -1))

    r_rot = torch.atan2(vx, vy)
    radius = batch_states[:, 4].reshape((batch, -1))

    vx1 = (batch_states[:, 11] * torch.cos(rot) + batch_states[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (batch_states[:, 12] * torch.cos(rot) - batch_states[:, 11] * torch.sin(rot)).reshape((batch, -1))
    h_rot = torch.atan2(vx1, vy1)
    # print(h_rot - r_rot)
    hr_rot = h_rot - r_rot
    vh = torch.norm(torch.cat([vx1, vy1], dim=1), 2, dim=1, keepdim=True)
    px1 = (batch_states[:, 9] - batch_states[:, 0]) * torch.cos(rot) + \
          (batch_states[:, 10] - batch_states[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (batch_states[:, 10] - batch_states[:, 1]) * torch.cos(rot) - \
          (batch_states[:, 9] - batch_states[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = batch_states[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1

    da = torch.norm(torch.cat([(batch_states[:, 0] - batch_states[:, 9]).reshape((batch, -1)),
                               (batch_states[:, 1] - batch_states[:, 10]).reshape((batch, -1))],
                              dim=1), 2, dim=1, keepdim=True)
    da=torch.min(da).repeat(5).view(5,1)

    new_state = torch.cat([radius, vx, vy, dg, rot_g, r_rot, radius_sum, hr_rot, da, px1, py1, vx1, vy1, radius1], dim=1)

    torch.set_printoptions(linewidth=200)

    return new_state


def rotate(state):
    """
    Transform the coordinate to agent-centric.
    Input state tensor is of size (batch_size, state_length)
    """
    D = []

    # old_state = rotate_s(policy_config, state)
    # print(old_state[:, 2:])

    def dist(human):
        # sort human order by decreasing distance to the robot
        # print((np.linalg.norm(np.array(human.position) - np.array(state.self_state.position)))*10)
        dist = np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))
        D.append(dist)

        return dist

    state.human_states = sorted(state.human_states, key=dist, reverse=True)
    # current_dist_weight = policy_config.getfloat("sac_rl", "current_dist_weight")

    # def dist(human):
    #     # sort human order by decreasing distance to the robot
    #     current_dist = np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))
    #     rpx, rpy = state.self_state.position
    #     fhx, fhy = (
    #         0.25 * human.vx + human.px,
    #         0.25 * human.vy + human.py,
    #     )
    #     next_possible_dist = np.linalg.norm(np.array([rpx, rpy]) - np.array([fhx, fhy]))
    #     D.append(next_possible_dist)
    #     return current_dist_weight * current_dist + (1 - current_dist_weight) * next_possible_dist

    # r0 = 0.3
    # r1 = 0.7
    # m = 0.55
    # v = []
    # r = []
    # R = []
    # theta = []
    # alphas = []
    # dz_Angle = []
    # human_Angle = []
    # dz_Distance = []
    # dangerous_value = []
    # for i in range(len(state.human_states)):
    #     # calculate geometric factor
    #     v.append(np.linalg.norm(state.human_states[i].velocity))
    #     r.append(r0 + r1 + m * v[i])
    #     theta.append((11 * pi / 6) * pow(e, -3 * v[i]) + pi / 6)
    #     # R.append(r[k] + (r[k] * self.r0 / (r[k] * sin(theta[k] / 2) - self.r0)))
    #     alpha = theta[i] / 2 - math.asin(r0 / r[i])
    #     alphas.append(alpha)
    #     R.append(sqrt(r[i] ** 2 - r0 ** 2) + r0 / tan(alphas[i]))
    #     vx, vy = state.human_states[i].velocity
    #     human_Angle.append(atan2(vy, vx))
    #     dz_x = state.human_states[i].px - r0 / sin(alphas[i]) * cos(human_Angle[i])
    #     dz_y = state.human_states[i].py - r0 / sin(alphas[i]) * sin(human_Angle[i])
    #     dz_Angle.append(atan2(dz_y - state.self_state.py, dz_x - state.self_state.px))
    #     dz_Distance.append(sqrt(pow(state.self_state.px - dz_x, 2) + pow(state.self_state.py - dz_y, 2)))
    #     # calculate dangerous_value
    #     angle_diss = abs(dz_Angle[i] - human_Angle[i])
    #     if angle_diss <= alphas[i] and (r0 / sin(alphas[i]) + r0) < dz_Distance[i] < R[i]:
    #         dangerous_value.append(cos(dz_Angle[i] - human_Angle[i]) * (1 - dz_Distance[i] / R[i]))
    #     elif angle_diss > alphas[i] and (r0 / sin(alphas[i]) + r0) < dz_Distance[i] < R[i]:
    #         dangerous_value.append(-abs(cos(dz_Angle[i] - human_Angle[i]) * (1 - dz_Distance[i] / R[i])))
    #     else:
    #         dangerous_value.append(-abs(dz_Distance[i]))
    #
    # # sort human order by decreasing dangerous_value to the robot
    # combination = zip(state.human_states, dangerous_value)
    # # dang = sorted(combination, key=lambda x: x[-1], reverse=True)
    # # dan=np.delete(dang,-2,axis=1)
    # human_states = sorted(combination, key=lambda x: x[-1], reverse=True)
    # human_states = np.delete(human_states, -1, axis=1)
    # state.human_states = human_states.tolist()
    # state.human_states = sum(state.human_states, [])
    new_state = rotate_s(state)

    return new_state


class MeanStdevFilter:
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D crossTF99.6%
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean ** 2,
                self.eps
            ))

    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.flag = True

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if len(self.buffer) >= self.capacity and self.flag is True:
            print('Replay experience pool is full')
            self.flag = False
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            lambda x: np.stack([i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in x]), zip(*batch))

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.buffer = []
        self.position = 0
        self.flag = True
        self.frame = 1
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_frames = 100000

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # if len(self.buffer) >= self.capacity and self.flag is True:
        #     print('Replay experience pool is full')
        #     self.flag = False
        # print(self.buffer)
        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1
        # print(bool(self.buffer))
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new crossTF99.6% on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.priorities[self.position] = max_prio
        # print(max_prio)

    def sample(self, batch_size):
        N = len(self.buffer)

        if N == self.capacity:
            prios = self.priorities
        else:
            # print(self.position)
            prios = self.priorities[:self.position]
        # print(prios)
        probs = prios ** self.alpha
        P = probs / probs.sum()
        indices = np.random.choice(N, batch_size, p=P)
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = map(
            lambda x: np.stack([i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in x]), zip(*samples))

        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            # print(prio)
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)


class ReplayPool:
    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim)),
            'reward': np.zeros(self.capacity),
            'nextstate': np.zeros((self.capacity, self._state_dim)),
            'real_done': np.zeros(self.capacity, dtype='bool')
        }

    def push(self, transition: Transition):
        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        # num_samples = transition.rhh_hxs.shape[0]
        # print(num_samples)
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity
        # print(idx)

        for key, value in transition._asdict().items():
            # print(key, value)
            self._memory[key][idx] = value
        # print(self._memory['rnn_hxs'])
        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k, v in self._memory.items()}
        return Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) \
            if not unique else self._rng.choice(self._size, size=batch_size, replace=False)

        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(),AffineTransform(loc=-1,scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as
        # it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        return 2. * (math.log(2.) - x - softplus(-2. * x))


def make_checkpoint(agent, step_count, env_name):
    q_funcs, target_q_funcs, policy, log_alpha = agent.q_funcs, agent.target_q_funcs, \
        agent.policy, agent.log_alpha

    save_path = "checkpoints/model-{}.pt".format(step_count)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'log_alpha_state_dict': log_alpha
    }, save_path)
