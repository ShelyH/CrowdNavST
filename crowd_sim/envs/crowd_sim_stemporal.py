#!/usr/bin/env python2
# coding=utf-8
import os
import sys
import numpy as np
from collections import deque

import torch
from gym import spaces
from numpy.linalg import norm
from crowd_nav.utils.info import *
from crowd_nav.utils.utils_sac import rotate
from crowd_sim.envs import CrowdSim
from crowd_sim.envs.utils.human import Human
from math import pow, pi, sqrt, atan2, sin, cos, e

from crowd_sim.envs.utils.state import JointState

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class CrowdSim_Stemporal(CrowdSim):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.frames_num = 5
        self.time_frames = deque(maxlen=self.frames_num)

    def configure(self, config):
        self.config = config
        self.disfromhuman = []
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.ros_visualize = config.getboolean('env', 'ros_visualize')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.robot_scan_radius = config.getfloat("robot", "scan_radius")
        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.getboolean('env', 'random_goal_changing')
        if self.random_goal_changing:
            self.goal_change_chance = config.getfloat('env', 'goal_change_chance')
        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.getboolean('env', 'end_goal_changing')
        if self.end_goal_changing:
            self.end_goal_change_chance = config.getfloat('env', 'end_goal_change_chance')
        # configure randomized radii changing when reaching goals
        self.random_radii = config.getboolean('env', 'random_radii')
        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.getboolean('env', 'random_v_pref')

        if self.config.get('humans', 'policy') == 'orca' \
                or self.config.get('humans', 'policy') == 'social_force':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000,
                              'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
            self.group_human = config.getboolean('env', 'group_human')
        else:
            raise NotImplementedError
        a_high = np.array([np.inf] * 2)
        self.action_space = spaces.Box(-a_high, a_high, dtype=np.float64)
        s_high = np.array([np.inf] * (5 * self.human_num + 9))
        self.state_space = spaces.Box(-s_high, s_high, dtype=np.float64)

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        # print 'env reset in crow_sim'
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        if test_case is not None:
            self.case_counter[phase] = test_case

        self.global_time = 0
        self.human_times = [0] * self.human_num
        self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, pi / 2)

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    # human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=self.human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase] test_size = 500
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                # assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 5
                    self.humans = [Human(self.config, 'humans')
                                   for _ in range(self.human_num)]
                    self.humans[0].set(2, -3, -2, 3, 0, 0, np.pi / 2)
                    self.humans[1].set(3.8, -2.3, -3.8, 2.3, 0, 0, np.pi / 2)
                    self.humans[2].set(-4, 0.2, 4, -0.2, 0, 0, np.pi / 2)
                    self.humans[3].set(4, 0.8, -4, -0.8, 0, 0, np.pi / 2)
                    self.humans[4].set(-2.6, -2.6, 2.6, 2.6, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # state = [human.get_observable_state() for human in self.humans]
        # state += [self.robot.get_full_state()]

        ob = [human.get_observable_state() for human in self.humans]
        self.potential = abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) -
                                            np.array([self.robot.gx, self.robot.gy])))
        time_frames_state = JointState(self.robot.get_full_state(), ob)
        time_frames_state = rotate(time_frames_state)
        self.time_frames.clear()
        for i in range(self.frames_num):
            self.time_frames.append(time_frames_state)
        # np.set_printoptions(linewidth=200)
        time_frames_state = torch.tensor(
            np.concatenate(self.time_frames).reshape((self.frames_num * self.human_num, -1)))
        # print(time_frames_state.shape)
        return time_frames_state

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision,
        update environment and return (ob, reward, done, info)
        # """
        robot_v_prev = np.array([self.robot.vx, self.robot.vy])
        action = self.robot.clip_action(action, self.robot.v_pref, robot_v_prev, self.time_step)
        # global ob
        # print(action)
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans
                  if other_human != human]

            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_action = human.act(ob, False)
            # human_action = self.robot.clip_action(human_action, self.robot.v_pref, robot_v_prev, self.time_step)
            # print(human_action)
            human_actions.append(human_action)
        self.human_actions = human_actions

        # collision detection
        dmin = float('inf')
        collision = False

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            # if closest_dist < self.discomfort_dist:
            #     danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        px, py = self.robot.compute_position(action, self.time_step)
        end_position = np.array([px, py])

        goal_vec = np.array(self.robot.get_goal_position()) - end_position
        goal_dist = norm(goal_vec)

        reaching_goal = goal_dist <= self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            potential_cur = np.linalg.norm(end_position - np.array(self.robot.get_goal_position()))
            reward = 2 * (self.potential - potential_cur) - potential_cur * 0.1
            self.potential = potential_cur
            done = False
            info = Nothing()
        if update:
            self.states.append([self.robot.get_full_state(), [
                human.get_full_state() for human in self.humans]])
            self.previous_humans = self.humans

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step

            # Update a specific human's goal once its reached its original goal
            if self.end_goal_changing:
                for human in self.humans:
                    if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                        self.update_human_goal(human)

            # Update all humans' goals randomly midway through episode
            if self.random_goal_changing:
                if self.global_time % 5 == 0:
                    self.update_human_goals_randomly()

            # compute the time sequence observation
            ob = [human.get_observable_state() for human in self.humans]
            time_frames_state = JointState(self.robot.get_full_state(), ob)
            time_frames_state = rotate(time_frames_state)
            self.time_frames.append(time_frames_state)
            # np.set_printoptions(linewidth=200)
            time_frames_state = torch.tensor(
                np.concatenate(self.time_frames).reshape((self.frames_num * self.human_num, -1)))
        else:
            pass

        return time_frames_state, reward, done, info
