#!/usr/bin/env python2
# coding=utf-8
import tf
import logging
import math
import os
import sys

import gym
import numpy as np
import rvo2
import torch
from gym import spaces
import rospy
from numpy.linalg import norm

from crowd_nav.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import FullState
from crowd_sim.envs.utils.utils import point_to_segment_dist
from math import pow, pi, sqrt, atan2, sin, cos, e

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist, Point, Pose, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

kp_distance = 0.7
ki_distance = 0.002
kd_distance = 0.1

kp_angle = 3
ki_angle = 0.002
kd_angle = 0.5

UNIT_distance = 0.3


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.reset_proxy = None
        self.odom_frame = None
        self.tf_listener = None
        self.dis = None
        self.disfromhuman = None
        self.robot_scan_radius = None
        self.random_v_pref = None
        self.random_radii = None
        self.random_goal_changing = None
        self.potential = None
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        # success_reward = 1
        self.success_reward = None
        # collision_penalty = -0.25
        self.collision_penalty = None
        # discomfort_dist = 0.5
        self.discomfort_dist = None
        # discomfort_penalty_factor = 0.5
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.base_frame = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.human_actions = None
        self.previous_humans = None
        self.ax = None
        self.fig = None
        # self.map = gridmap(xlimit=12, ylimit=12, resolution=0.02, time_step=0.25)

    def init_ros(self):
        rospy.init_node('crowd')
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        position = Point()
        move_cmd = Twist()
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint',
            rospy.Time(), rospy.Duration(10))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link',
                rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Execption")

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
        # print(config)
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

        if self.config.get('humans', 'policy') == 'orca':
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
            # self.action_space=None
        else:
            raise NotImplementedError

        a_high = np.array([np.inf] * 2)
        self.action_space = spaces.Box(-a_high, a_high, dtype=np.float64)
        s_high = np.array([np.inf] * (5 * self.human_num + 9))
        self.state_space = spaces.Box(-s_high, s_high, dtype=np.float64)

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if not self.group_human:  # to improve the runtime
                    humans_copy = []
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)

                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases' robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False
                    # print(self.group_human)  # False
                    if self.group_human:
                        pass
                    else:
                        for agent in [self.robot] + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                    if not collide:
                        break

                # Give human new goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human):
        # Update human goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            if not self.group_human:
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                human.v_pref += np.random.uniform(-0.1, 0.1)

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                if self.group_human:  # False
                    pass
                else:
                    for agent in [self.robot] + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break
            # _, gx, gy = self.generate_circle_crossing_human()
            # Give human new goal
            human.gx = gx
            human.gy = gx
        return

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                human = self.generate_circle_crossing_human()
                self.humans.append(human)
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 1 else False
            prob = np.random.random()
            for key, value in sorted(
                    static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                px, py = -4, -1
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        # px += 3
                        py = (np.random.random() - 0.5) * 6
                        # py = (np.random.random()) * 10 * 0
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((
                                    px - agent.px,
                                    py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 0:
                        human = self.generate_circle_crossing_human()
                    else:
                        # print(True)
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)
        # print(self.px, self.py, gx, gy)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = np.random.random() * self.square_width * 0.5 * sign
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx,
                         gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        # print(px, py, gx, gy)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        # params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, 10, 10, 5, 5, 0.3, 1)
        sim.addAgent(self.robot.get_position(), 10, 10, 5, 5, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), 10, 10, 5, 5, human.radius, human.v_pref,
                         human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append(
                [self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

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
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            # print(self.robot.policy.multiagent_training)
            self.human_times = [0] * (
                self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}  # test: 1000
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, pi / 2)

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    # print(human_num)
                    self.generate_random_human_position(human_num=human_num,
                                                        rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num,
                                                        rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase] test_size = 500
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 5
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -5, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                    self.humans[3].set(-5, 5, 5, -5, 0, 0, np.pi / 2)
                    self.humans[4].set(-5, 0, 5, -5, 0, 0, np.pi / 2)
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
        # print(self.humans)
        ob = [human.get_observable_state() for human in self.humans]
        # print(np.arctan2(self.robot.vy, self.robot.vx))
        self.last_alpha = np.arctan2(self.robot.vy, self.robot.vx)
        self.potential = abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) -
                                            np.array([self.robot.gx, self.robot.gy])))
        if phase == 'test' and self.ros_visualize == True:
            self.init_ros()
            self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
            self.cmd_vel.publish(Twist())
            rospy.wait_for_service('gazebo/reset_simulation')
            try:
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print("gazebo/reset_simulation service call failed")
            self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            obstacle = []
            # init obstacle position
            for i in range(len(self.humans)):
                for j in range(len(model.name)):
                    if model.name[j] == 'Dynamic_obstacle_' + str(i):
                        obstacle.append(ModelState())
                        obstacle[i].model_name = model.name[j]
                        obstacle[i].pose = model.pose[j]
                        obstacle[i].pose.position.x = UNIT_distance * ob[i].px
                        obstacle[i].pose.position.y = UNIT_distance * ob[i].py
                        self.pub_model.publish(obstacle[i])
                # print("%d:", i, ob[i].px, ob[i].py)
        return ob

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        return (Point(*trans), rotation[2])

    def control(self, x, y):
        self.init_ros()
        (position, rotation) = self.get_odom()
        last_rotation = 0
        (goal_x, goal_y) = (float(x), float(y))

        goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
        now_distance = goal_distance

        move_cmd = Twist()
        r = rospy.Rate(50)

        previous_distance = 0
        total_distance = 0

        previous_angle = 0
        total_angle = 0
        obstacle = []
        state = [0, 0, 0, 0, 0]

        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)

        for i in range(len(self.humans)):
            for j in range(len(model.name)):
                if model.name[j] == 'Dynamic_obstacle_' + str(i):
                    obstacle.append(ModelState())
                    obstacle[i].model_name = model.name[j]
                    obstacle[i].pose = model.pose[j]
                    obstacle[i].pose.position.x = UNIT_distance * self.previous_humans[i].px
                    obstacle[i].pose.position.y = UNIT_distance * self.previous_humans[i].py
                    self.pub_model.publish(obstacle[i])

        while now_distance > 0.01:
            x_start = position.x
            y_start = position.y
            path_angle = atan2(goal_y - y_start, goal_x - x_start)

            rot_angle = path_angle - rotation
            if rot_angle > pi:
                rot_angle = rotation - 2 * pi
            elif rot_angle < -pi:
                rot_angle = rotation + 2 * pi

            diff_angle = rotation - previous_angle
            diff_distance = now_distance - previous_distance

            control_signal_distance = abs(1 - abs(
                rot_angle / 3.2)) * kp_distance * now_distance + ki_distance * total_distance + kd_distance * diff_distance
            control_signal_angle = kp_angle * rot_angle + ki_angle * total_angle + kd_angle * diff_angle

            move_cmd.linear.x = min(control_signal_distance, 0.4)
            move_cmd.angular.z = control_signal_angle

            if move_cmd.angular.z > 0:
                move_cmd.angular.z = min(move_cmd.angular.z, 2.0)
            else:
                move_cmd.angular.z = max(move_cmd.angular.z, -2.0)

            self.cmd_vel.publish(move_cmd)
            previous_angle = rotation
            previous_distance = now_distance
            total_distance = total_distance + previous_distance
            (position, rotation) = self.get_odom()
            now_distance = sqrt(pow((goal_x - position.x), 2) + pow((goal_y - position.y), 2))

            for i in range(len(self.humans)):
                for j in range(len(model.name)):
                    if model.name[j] == 'Dynamic_obstacle_' + str(i):
                        if state[i] == 0:
                            obstacle.append(ModelState())
                            obstacle[i].model_name = model.name[j]
                            obstacle[i].pose = model.pose[j]
                            obstacle[i].pose.position.x += UNIT_distance * 0.00112 * self.human_actions[i].vx
                            obstacle[i].pose.position.y += UNIT_distance * 0.00112 * self.human_actions[i].vy
                            self.pub_model.publish(obstacle[i])
                            if abs(obstacle[i].pose.position.x - UNIT_distance * self.humans[i].px) < 0.0001 and abs(
                                    obstacle[i].pose.position.y - UNIT_distance * self.humans[i].py) < 0.0001:
                                state[i] = 1
                                # print('state', i)
            r.sleep()

        (position, rotation) = self.get_odom()
        logging.info(
            "Current position and rotation are:{0} {1} {2} ".format(position.x, position.y, rotation))
        print("reached: ^_^")

        # rospy.loginfo("Stopping the robot ...")
        self.cmd_vel.publish(Twist())
        return

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision,
        update environment and return (ob, reward, done, info)
        """
        robot_v_prev = np.array([self.robot.vx, self.robot.vy])
        action = self.robot.clip_action(action, self.robot.v_pref, robot_v_prev, self.time_step)
        # global ob
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans
                  if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob, False))
        self.human_actions = human_actions
        # print(self.human_actions)
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
        # print(self.robot.vx, self.robot.vy)
        # print(dis)

        px, py = self.robot.compute_position(action, self.time_step)
        end_position = np.array([px, py])
        # print(end_position)
        goal_vec = np.array(self.robot.get_goal_position()) - end_position
        goal_dist = norm(goal_vec)
        if self.robot.kinematics == 'holonomic':
            r_v = np.linalg.norm(action)
            # print(np.arctan2(self.robot.vy, self.robot.vx))
            self.alpha = np.arctan2(self.robot.vy, self.robot.vx)
            # print(self.last_alpha, self.alpha, self.alpha - self.last_alpha)
            theta = abs(self.alpha - self.last_alpha)
        else:
            r_v = action.v
            self.alpha = action.r
            theta = abs(self.alpha)
        self.last_alpha = self.alpha
        # rg_dist = norm(end_position - np.array(self.robot.get_goal_position()))
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
            v_reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            # if theta > 1:
            #     rotate_reward = -theta * 2
            # else:
            #     rotate_reward = theta * 2
            reward = v_reward
            done = False
            info = Danger(dmin)
            # print(v_reward, theta, rotate_reward)
        # elif theta > 1:
        #     reward = -theta
        #     done = False
        #     info = Danger(theta)
        else:
            if theta > 1:
                rotate_reward = -theta
            else:
                rotate_reward = 0
            # print(theta)
            # print([self.robot.px, self.robot.py], end_position)

            # print('rotate_reward', rotate_reward)
            # potential_cur = np.linalg.norm(np.array([self.robot.px, self.robot.py])
            #                                - np.array(self.robot.get_goal_position()))
            potential_cur = np.linalg.norm(end_position - np.array(self.robot.get_goal_position()))
            # reward = 2 * (self.potential - potential_cur) - potential_cur * 0.2
            reward = 2 * (self.potential - potential_cur)
            # reward = -0.1
            # print(reward)
            # reward = - potential_cur * 0.2
            # print(- potential_cur * 0.2)
            # print([- potential_cur, danger_reward])
            # reward = -1
            if self.robot.kinematics == 'unicycle':
                # add a rotational penalty
                r_spin = -2 * action.r ** 2

                # add a penalty for going backwards
                if action.v < 0:
                    r_back = -2 * abs(action.v)
                else:
                    r_back = 0.
                # print(action.v, action.r, r_back, potential_cur)
                reward = reward + r_spin + r_back
            self.potential = abs(potential_cur)
            done = False
            info = Nothing()
        # scanned_humans, indexes = self.get_scanned_humans()
        # print(scanned_humans)
        if update:
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            self.previous_humans = self.humans
            # store state, action value and attention weights
            # self.states.append(
            #     [self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            # for i, human in enumerate(self.humans):
            #     # only record the first time the human reaches the goal
            #     if self.human_times[i] == 0 and human.reached_destination():
            #         self.human_times[i] = self.global_time
            #
            # Update a specific human's goal once its reached its original goal
            if self.end_goal_changing:
                for human in self.humans:
                    if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                        self.update_human_goal(human)

            # Update all humans' goals randomly midway through episode
            if self.random_goal_changing:
                if self.global_time % 5 == 0:
                    self.update_human_goals_randomly()

            # compute the observation
            ob = [human.get_observable_state() for human in self.humans]
            state = [human.get_observable_state() for human in self.humans]
            state += [self.robot.get_full_state()]

            # print(self.robot.get_full_state())
            if self.ros_visualize:
                px, py = self.robot.get_position()
            self.control(UNIT_distance * px, UNIT_distance * py)

        else:
            # Update a specific human's goal once its reached its original goal
            if self.end_goal_changing:
                for human in self.humans:
                    if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                        self.update_human_goal(human)

            # Update all humans' goals randomly midway through episode
            if self.random_goal_changing:
                if self.global_time % 5 == 0:
                    self.update_human_goals_randomly()

            ob = [human.get_next_observable_state(action) for human, action in
                  zip(self.humans, human_actions)]
            state = []
        # print(action, reward)
        # ob = self.transform_observation(self.robot, ob)
        # print('obs:', obs)
        return ob, reward, done, info, state

    def render(self, update=None):
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
        from matplotlib import animation, patches
        if update:
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            x_offset = 0.1
            y_offset = 0.1
            robot_color = 'yellow'
            goal_color = 'red'
            arrow_color = 'red'
            arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
            plt.rc('font', family='Times New Roman', size=20)
            cmap = plt.cm.get_cmap('hsv', 10)

            plt.rcParams["font.family"] = "Times New Roman"
            max_edge_width = 30
            alpha = 0.2
            edge_color = 'r'
            init_edge_width = 0.05
            artists = []

            # add goal
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color,
                                 marker='*', linestyle='None', markersize=16, label='Goal')
            self.ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX, robotY = self.robot.get_position()

            robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
            # add humans and change the color of them based on visibility
            human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in
                             self.humans]
            self.ax.add_artist(robot)
            artists.append(robot)
            # legend_font = {'size': 16, "family":'Times New Roman'}
            plt.legend([robot, goal], ['Robot', 'Goal'])

            # compute orientation in each step and add arrow to show the direction
            radius = self.robot.radius
            arrowStartEnd = []

            robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' \
                else np.arctan2(self.robot.vy, self.robot.vx)
            # print(rob)
            arrowStartEnd.append(((robotX, robotY),
                                  (robotX + radius * np.cos(robot_theta),
                                   robotY + radius * np.sin(robot_theta))))

            # arrowStartEnd = [((state[0].px, state[0].py),
            #                   (state[0].px + radius * np.cos(state[0].theta),
            #                    state[0].py + radius * np.sin(state[0].theta))) for state in self.states]
            # arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
            #           for orientation in arrowStartEnd]
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))] for i in
                               range(len(self.states))]
            # print(len(self.states))
            for k in range(len(self.states)):
                # print(robot_positions,human_positions)
                d_to = []
                for j in range(self.human_num):
                    h_r = np.array(np.array(robot_positions[k]) - np.array(human_positions[k][j]))
                    dis = norm(h_r)
                    d_to.append(dis)

                self.disfromhuman.append(d_to)
                edges_to_humans = [plt.Line2D([robot_positions[k][0] + x_offset, human_positions[k][j][0]],
                                              [robot_positions[k][1] + y_offset, human_positions[k][j][1]],
                                              linestyle='--',
                                              linewidth=init_edge_width * max_edge_width,
                                              color=edge_color, alpha=alpha) for j in range(len(self.humans))]

                edges = edges_to_humans

                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color='black', ls='solid')
                    human_directions = [
                        plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                   color=cmap(i), ls='solid')
                        for i in range(self.human_num)
                    ]
                    self.ax.add_artist(nav_direction)
                    artists.append(nav_direction)
                    # for human_direction in human_directions:
                    #     self.ax.add_artist(human_direction)
                    #     artists.append(human_direction)
            # add time annotation
            # print('')
            # del self.disfromhuman[:]
            for i, edge in enumerate(edges):
                self.ax.add_artist(edge)
                artists.append(edge)

            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=20)
            self.ax.add_artist(time)
            artists.append(time)
            time.set_text('Time: {:.2f}'.format(len(self.states) * self.time_step))

            dis = [
                plt.text(6.4, 3.5 - i, '{}: {:.3f}'.format(i, self.disfromhuman[0][i]),
                         fontsize=12) for i in range(len(self.humans))
            ]
            # print(self.disfromhuman)
            for i, human in enumerate(self.humans):
                # for edge in edges:
                #     edge.remove()
                theta = np.arctan2(human.vy, human.vx)
                arrowStartEnd.append(((human.px, human.py),
                                      (human.px + radius * np.cos(theta),
                                       human.py + radius * np.sin(theta))))

                dis[i].set_text('{}: {:.3f}'.format(i, self.disfromhuman[-1][i]))

            arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                      for arrow in arrowStartEnd]

            for arrow in arrows:
                self.ax.add_artist(arrow)
                artists.append(arrow)

            for i in range(len(self.humans)):
                self.ax.add_artist(human_circles[i])
                artists.append(human_circles[i])

                human_circles[i].set_color(c='r')
                plt.text(self.humans[i].px, self.humans[i].py, str(i),
                         color='black', fontsize=16)

            # print(self.disfromhuman)
            plt.pause(0.15)

            for item in artists:
                # print(item)
                item.remove()  # there should be a better way to do this. For example,
                # initially use add_artist and draw_artist later on
            for t in self.ax.texts:
                t.set_visible(False)
            # del self.disfromhuman[:]
            # self.disfromhuman.clear()
            # plt.show()
        else:
            fig, ax = plt.subplots(figsize=(8.5, 7.5))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=20)
            ax.set_ylabel('y(m)', fontsize=20)
            plt.figure(1, constrained_layout=True)
            plt.rc('font', family='Arial', size=20)
            plt.rcParams["font.family"] = "Arial"
            # ax.spines['top'].set_color('none')
            # ax.spines['right'].set_color('none')
            plt.yticks(fontproperties='Arial', size=20)
            plt.xticks(fontproperties='Arial', size=20)
            # set(gca, 'xtickmode', 'manual', 'ytickmode', 'manual', 'ztickmode', 'manual')

            x_offset = 0.11
            y_offset = 0.11
            cmap = plt.cm.get_cmap('hsv', 10)
            robot_color = 'yellow'
            goal_color = 'red'
            arrow_color = 'red'
            arrow_style = patches.ArrowStyle("->", head_length=5, head_width=4)

            max_edge_width = 30
            alpha = 0.2
            edge_color = 'red'
            init_edge_width = 0.05
            artists = []
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))] for i in
                               range(len(self.states))]
            # print(self.human_num)
            for i in range(len(self.states)):
                if i % 4 == 0 or i == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[i], self.robot.radius, fill=True, linewidth=1, color=robot_color)
                    humans = [plt.Circle(human_positions[i][k], self.humans[k].radius, fill=False, linewidth=1.5,
                                         color=cmap(k))
                              for k in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                global_time = i * self.time_step
                if global_time % 4.2 == 0 or i == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[k].center[0] - x_offset, agents[k].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for k in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                d_to = []
                for j in range(self.human_num):
                    h_r = np.array(np.array(robot_positions[i]) - np.array(human_positions[i][j]))
                    dis = norm(h_r)
                    d_to.append(dis)
                    # print(d_to)
                self.disfromhuman.append(d_to)

                if i != 0:
                    nav_direction = plt.Line2D((self.states[i - 1][0].px, self.states[i][0].px),
                                               (self.states[i - 1][0].py, self.states[i][0].py), color='yellow',
                                               ls='solid')
                    ax.add_artist(nav_direction)
                    human_directions = [
                        plt.Line2D((self.states[i - 1][1][k].px, self.states[i][1][k].px),
                                   (self.states[i - 1][1][k].py, self.states[i][1][k].py),
                                   color=cmap(k), ls='solid')
                        for k in range(self.human_num)
                    ]
                    ax.add_artist(nav_direction)
                    artists.append(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
                        artists.append(human_direction)

            time = plt.text(8.8, 8, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)
            artists.append(time)
            # artists.append(time)
            time.set_text('Time: {:.2f}'.format(len(self.states) * self.time_step))
            # add robot and its goal
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None',
                                 markersize=14, label='Goal')

            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            artists.append(robot)
            ax.add_artist(goal)
            artists.append(goal)

            # add humans and their numbers
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i)) for i in
                      range(len(self.humans))]
            human_numbers = [
                plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i), color='black',
                         fontsize=16) for i in range(len(self.humans))]
            plt.legend([robot, goal], ['Robot', 'Goal'], bbox_to_anchor=(1, 1))
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])
                artists.append(human)
                artists.append(human_numbers[i])

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (
                    state[0].px + radius * np.cos(state[0].theta), state[0].py + radius * np.sin(state[0].theta)))
                               for state in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        # print(theta, agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py),
                                            (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            global ar
            ar = arrows
            for arrow in arrows:
                ax.add_artist(arrow)
                artists.append(arrow)

            # print(self.humans)
            edges_to_humans = [plt.Line2D([robot_positions[0][0], human_positions[0][i][0]],
                                          [robot_positions[0][1], human_positions[0][i][1]], linestyle='--',
                                          linewidth=init_edge_width * max_edge_width, color=edge_color,
                                          alpha=alpha) for i in range(len(self.humans))]

            edges = edges_to_humans
            # for i, edge in enumerate(edges):
            #     ax.add_artist(edge)
            dis = [
                plt.text(8, 8 - i, '{}: {:.3f}'.format(i, self.disfromhuman[0][i]),
                         fontsize=12) for i in range(len(self.humans))
            ]

            def Update(frame_num):
                global ar
                nonlocal edges
                nonlocal edge_color
                nonlocal alpha
                nonlocal max_edge_width
                robot.center = robot_positions[frame_num]
                # print(frame_num)
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))

                    for arrow in ar:
                        arrow.remove()
                    ar = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                  arrowstyle=arrow_style)
                          for orientation in orientations]
                    for arrow in ar:
                        ax.add_artist(arrow)
                        artists.append(arrow)
                    dis[i].set_text('{}: {:.3f}'.format(i, self.disfromhuman[frame_num][i]))
                # add time annotation
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                # plt.show()

                # for edge in edges:
                #     edge.remove()

                edges_to_humans = [plt.Line2D([robot_positions[frame_num][0], human_positions[frame_num][i][0]],
                                              [robot_positions[frame_num][1], human_positions[frame_num][i][1]],
                                              linestyle='--',
                                              linewidth=init_edge_width * max_edge_width,
                                              color=edge_color, alpha=alpha) for i in range(len(self.humans))]

                edges = edges_to_humans

                # for i, edge in enumerate(edges):
                #     ax.add_artist(edge)

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    # plt.pause(10)
                else:
                    # plt.pause(len(self.states * 2) * self.time_step)
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, Update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True
            # plt.close(fig)
            plt.show()

            # fig.clf()
            # plt.show(block=False)
            # plt.pause(len(self.states) * self.time_step + 0.6)
            # plt.close()
