import copy
import math
import time
from math import sqrt, atan2, pi

import numpy as np
import rospy
import tf
from crowd_sim.envs.utils.state import ObservableState, FullState
from message_filters import ApproximateTimeSynchronizer, Subscriber

from crowd_sim.envs.utils.human import Human
from gazebo_msgs.srv import GetModelState
from crowd_nav.utils.info import *
from std_srvs.srv import Empty
from numpy.linalg import norm
from sensor_msgs.msg import JointState
from crowd_sim.envs.crowd_sim import CrowdSim
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist, Point, Pose, Quaternion

from tf.transformations import euler_from_quaternion, quaternion_from_euler

UNIT_distance = 0.3


class Crowdsim2real_V2(CrowdSim):
    def __init__(self):
        super().__init__()
        self.last_left = 0.
        self.last_right = 0.
        self.last_w = 0.0
        self.odom_frame = 'odom'
        self.desiredVelocity = [0.0, 0.0]
        self.fixed_time_interval = 0.1
        self.ROSStepInterval = 0.03
        self.last_v = 0
        self.lastTime = 0
        self.speed = None
        self.potential = None
        # self.cmd_vel.publish(Twist())

    def configure(self, config):
        super().configure(config)
        rospy.init_node('crowd')
        jointStateSub = Subscriber("/joint_states", JointState)
        # subList = [jointStateSub]
        # self.ats = ApproximateTimeSynchronizer(subList, queue_size=1, slop=1)
        # self.ats.registerCallback(self.state_cb_dummy)
        self.human_radius = config.getfloat('humans', 'radius')
        self.robot_radius = config.getfloat('robot', 'radius')
        self.v_pref = config.getint('robot', 'v_pref')
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

    def state_cb_dummy(self, jointStateMsg):
        self.jointMsg = jointStateMsg

    def init_ros(self):
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.tf_listener = tf.TransformListener()
        # try:
        #     self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1))
        #     self.base_frame = 'base_footprint'
        # except (tf.Exception, tf.ConnectivityException, tf.LookupException):
        #     try:
        #         self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1))
        #         self.base_frame = 'base_link'
        #     except (tf.Exception, tf.ConnectivityException, tf.LookupException):
        #         rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
        #         rospy.signal_shutdown("tf Execption")

    def velocity_callback(self, data):
        self.linear_velocity = data.linear.x
        self.angular_velocity = data.angular.z

    def return_velocity(self):
        return self.linear_velocity, self.angular_velocity

    def smooth(self, v, w):
        beta = 0.1  # TODO: you use 0.2 in the simulator
        left = (2. * v - 0.23 * w) / (2. * 0.035)
        right = (2. * v + 0.23 * w) / (2. * 0.035)
        # print('199:', left, right)
        left = np.clip(left, -17.5, 17.5)
        right = np.clip(right, -17.5, 17.5)
        # print('202:', left, right)
        left = (1. - beta) * self.last_left + beta * left
        right = (1. - beta) * self.last_right + beta * right
        # print('205:', left, right)

        self.last_left = copy.deepcopy(left)
        self.last_right = copy.deepcopy(right)

        v_smooth = 0.035 / 2 * (left + right)
        w_smooth = 0.035 / 0.23 * (right - left)

        return v_smooth, w_smooth

    def get_odom(self):
        model_state = self.get_model_state('robot', 'world')

        px = model_state.pose.position.x
        py = model_state.pose.position.y
        return px, py

    # def get_robot_velocity(self):

    def get_human_states(self):
        obstacle = []
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        # print(model.name)
        human_states = []
        # obstacle.twist = Twist()
        for i in range(len(self.humans)):
            # for j in range(len(model.name)):
            if model.name[i] == 'human_' + str(i + 1):
                model_state = self.get_model_state('human_' + str(i + 1), 'world')

                px = model_state.pose.position.x
                py = model_state.pose.position.y
                vx = model_state.twist.linear.x
                vy = model_state.twist.linear.y
                # print(px, py, vx, vy)
                human_states.append(ObservableState(px, py, vx, vy, 0.4))
        return human_states

    def get_robot_states(self):
        # self.get_robot_velocity()
        # speed_msg = rospy.wait_for_message('/cmd_vel', Twist)
        # print(speed_msg)
        model_state = self.get_model_state('robot', 'world')

        px, py = self.get_odom()

        vx = model_state.twist.linear.x
        vy = model_state.twist.linear.y
        self.theta = pi / 2

        return FullState(px, py, vx, vy, self.robot_radius, self.gx, self.gy, self.v_pref, self.theta)

    def human_step(self, i, obstacle, action):
        """
        Perform an action and update the state
        """
        r = rospy.Rate(5000)
        # obstacle = []
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)

        # human_states = []
        v_linear = sqrt(action.vx ** 2 + action.vy ** 2)
        # v_angular = atan2(action.vy, action.vx) / (2 * pi)
        if model.name[i] == 'human_' + str(i + 1):
            # obstacle.append(ModelState())
            # print(model.name[j])
            obstacle[i].model_name = model.name[i]
            obstacle[i].pose = model.pose[i]
            # obstacle[i].twist = Twist()
            # print('human_' + str(i + 1), action.vx, action.vy)
            obstacle[i].twist.linear.x = action.vx
            obstacle[i].twist.linear.y = action.vy
            # obstacle[i].twist.angular.z = action.r
            self.humans[i].vx = action.vx
            self.humans[i].vy = action.vy
            self.humans[i].px = obstacle[i].pose.position.x
            self.humans[i].py = obstacle[i].pose.position.y
            self.pub_model.publish(obstacle[i])

            # state_human = self.get_model_state('human_' + str(i + 1), "world")
            rospy.sleep(0.04)
            r.sleep()

    def reset(self, phase='test', test_case=None):
        """
            Set px, py, gx, gy, vx, vy, theta for robot and humans
            :return:
        """
        # print 'env reset in crow_sim'
        self.init_ros()
        self.reset_proxy()
        self.currentTime = 0
        self.step_counter = 0
        self.human_times = [0] * self.human_num
        self.global_time = 0
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        if test_case is not None:
            self.case_counter[phase] = test_case

        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}  # test: 1000
        self.robot.set(0, -4, 0, 4, 0, 0, pi / 2)
        self.gx, self.gy = self.robot.get_goal_position()
        if self.case_counter[phase] >= 0:
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            if phase in ['train', 'val']:
                human_num = self.human_num if self.robot.policy.multiagent_training else 1
                # print(human_num)
                self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
            else:
                self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
            # case_counter is always between 0 and case_size[phase] test_size = 500
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        ob = [human.get_observable_state() for human in self.humans]
        # ob = self.get_human_states()
        # print(np.arctan2(self.robot.vy, self.robot.vx))
        self.last_alpha = np.arctan2(self.robot.vy, self.robot.vx)
        # print(self.robot.px, self.robot.py)
        self.potential = abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) -
                                            np.array([self.robot.gx, self.robot.gy])))
        # rospy.wait_for_service('gazebo/reset_simulation')
        r = rospy.Rate(500)
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for j in range(2):
            for i in range(self.human_num):
                if model.name[i] == 'human_' + str(i + 1):
                    # print(model.name[i])
                    obstacle.model_name = model.name[i]
                    obstacle.pose = model.pose[i]
                    obstacle.pose.position.x = ob[i].px
                    obstacle.pose.position.y = ob[i].py
                    self.pub_model.publish(obstacle)
                    rospy.sleep(0.008)

            r.sleep()
        return ob

    def step(self, action, update=True):
        robot_v_prev = np.array([self.robot.vx, self.robot.vy])
        action = self.robot.clip_action(action, self.robot.v_pref, robot_v_prev, self.time_step)
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans
                  if other_human != human]
            # print(ob[0])
            human_actions.append(human.act(ob, False))
        self.human_actions = human_actions
        # print(ob[0])
        realAction = Twist()
        # v_linear = sqrt(action.vx ** 2 + action.vy ** 2)
        # v_angular = atan2(action.vy, action.vx)
        # self.currentTime = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

        # self.delta_t = self.currentTime - self.lastTime

        # self.step_counter = self.step_counter + 1
        # self.lastTime = self.currentTime
        # self.global_time = self.global_time + self.delta_t
        # self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + v_linear, -self.robot.v_pref, self.robot.v_pref)
        # self.desiredVelocity[1] = v_angular / self.fixed_time_interval
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        # v_smooth, w_smooth = self.smooth(self.desiredVelocity[0], self.desiredVelocity[1])
        # self.last_v = v_smooth
        print(action)
        # realAction.linear.x = action.vx
        # realAction.linear.y = action.vy
        model_list = []
        for i in range(len(model.name)):
            model_list.append(ModelState())
            if model.name[i] == 'robot':
                model_list[i].model_name = model.name[i]
                model_list[i].pose = model.pose[i]
                # obstacle[i].twist = Twist()
                # print('human_' + str(i + 1), action.vx, action.vy)
                model_list[i].twist.linear.x = action.vx
                model_list[i].twist.linear.y = action.vy
                # model_list[i].twist.angular.z = action.r

                self.pub_model.publish(model_list[i])
        # realAction.angular.z = action.r
        # self.cmd_vel.publish(realAction)
        # time.sleep(0.03)
        dmin = float('inf')
        collision = False

        px, py = self.get_odom()
        # end_position = np.array([px, py])
        obstacle = []
        for i in range(self.human_num):
            # print(model.name[j])
            if model.name[i] == 'human_' + str(i + 1):
                obstacle.append(ModelState())
                # print(model.name[j])
                obstacle[i].model_name = model.name[i]
                obstacle[i].pose = model.pose[i]
                x = obstacle[i].pose.position.x
                y = obstacle[i].pose.position.y
                closest_dist = sqrt(
                    pow(x - px, 2) + pow(y - py, 2)) - self.human_radius * 2
                # print(closest_dist)
                if closest_dist < 0:
                    collision = True
                    # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist
        # print(gx, gy)
        obstacle = []
        # print('robot', action.v)
        # self.currentTime_else = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

        for i, human_action in enumerate(human_actions):
            obstacle.append(ModelState())
            self.human_step(i, obstacle, human_action)
        # reward = None
        ob = self.get_human_states()

        # time.sleep(0.1)
        rospy.wait_for_service("/gazebo/unpause_physics")
        px, py = self.get_odom()
        # get_v = self.get_model_state("turtlebot3_burger", "world")
        # print(get_v)
        goal_dist = sqrt(pow(self.gx - px, 2) + pow(self.gy - py, 2))
        # print(goal_dist)
        reaching_goal = goal_dist <= self.robot.radius + 0.2
        # if self.global_time >= self.time_limit:
        #     reward = 0
        #     done = True
        #     info = Timeout()
        if collision:
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
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor
            done = False
            info = Danger(dmin)
        else:
            potential_cur = goal_dist
            reward = 2 * (self.potential - potential_cur)
            self.potential = abs(potential_cur)
            done = False
            info = Nothing()

        return ob, reward, done, info
