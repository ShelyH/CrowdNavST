import copy
import math
import random
import time
import tf
import threading
import numpy as np
import rospy
from nav_msgs.msg import Path
from crowd_sim.envs.utils.state import ObservableState, FullState
from message_filters import ApproximateTimeSynchronizer, Subscriber
from crowd_sim.envs.utils.human import Human
from math import sqrt, atan2, pi
from gazebo_msgs.srv import GetModelState
from crowd_nav.utils.info import *
from std_srvs.srv import Empty
from numpy.linalg import norm
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from crowd_sim.envs.crowd_sim import CrowdSim
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist, Point, PoseStamped, Pose, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

UNIT_distance = 0.3


class Crowdsim2real(CrowdSim):
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
        self.use_amcl_pose = True
        rospy.init_node('crowd')
        self.lock = threading.Lock()
        # self.odom_sub = rospy.Subscriber("/current_pose", PoseStamped, self.odom_callback, queue_size=10)
        # print(self.odom_sub)
        self.info_robot_path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.robot_path = Path()
        self.robot_path.header.stamp = rospy.Time.now()
        self.robot_path.header.frame_id = 'map'
        ## Goal Marker
        self.info_goal_marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
        self.goal_marker = Marker()
        self.goal_info_marker = Marker()
        self.goal_info_marker.header.stamp = rospy.Time.now()
        self.goal_info_marker.action = self.goal_info_marker.ADD
        self.goal_info_marker.ns = 'basic_shape'
        self.goal_info_marker.type = self.goal_info_marker.TEXT_VIEW_FACING
        self.goal_info_marker.text = "Goal"
        self.goal_info_marker.type = self.goal_info_marker.CUBE
        self.goal_info_marker.header.frame_id = 'map'
        self.goal_info_marker.id = 0
        self.goal_info_marker.pose.position.x = 0
        self.goal_info_marker.pose.position.y = 4
        self.goal_info_marker.pose.position.z = 0
        # self.goal_info_marker.pose.position.z += 0.4 * 1.5

        self.goal_info_marker.pose.orientation.x = 0
        self.goal_info_marker.pose.orientation.y = 0
        self.goal_info_marker.pose.orientation.z = 0
        self.goal_info_marker.pose.orientation.w = 1
        self.goal_info_marker.scale.x = 0.4
        self.goal_info_marker.scale.y = 0.4
        self.goal_info_marker.scale.z = 0.4
        self.goal_info_marker.color.a = 1.0
        self.goal_info_marker.color.r = 1.0
        self.goal_info_marker.color.g = 0
        self.goal_info_marker.color.b = 0

        self.marker_pub = rospy.Publisher('/dynamic_obstacle_markers', Marker, queue_size=10)
        # self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
        # self.marker = Marker()
        # self.marker.header.frame_id = "map"  # The marker will be in the "world" coordinate frame
        # self.marker.header.stamp = rospy.Time.now()
        # self.marker.ns = "obstacles"
        # self.marker.id = 0

        # self.marker.type = Marker.SPHERE
        # self.marker.action = Marker.ADD

    def configure(self, config):
        super().configure(config)
        jointStateSub = Subscriber("/joint_states", JointState)
        subList = [jointStateSub]
        self.ats = ApproximateTimeSynchronizer(subList, queue_size=1, slop=1)
        self.ats.registerCallback(self.state_cb_dummy)
        self.human_radius = config.getfloat('humans', 'radius')
        self.robot_radius = config.getfloat('robot', 'radius')
        self.v_pref = config.getint('robot', 'v_pref')
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=4)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

    def state_cb_dummy(self, jointStateMsg):
        self.jointMsg = jointStateMsg

    def init_ros(self):
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.tf_listener = tf.TransformListener()
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Execption")

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
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            # print(trans)
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        return Point(*trans)

    def get_robot_pose(self):
        odom = self.get_model_state("turtlebot3_waffle", "world")
        self.robot_path.poses.append(odom)
        self.info_robot_path_pub.publish(self.robot_path)
        # rospy.sleep(0.03)

        # self.robot_path.poses.append(odom)
        # self.info_robot_path_pub.publish(self.robot_path)
        # rospy.sleep(0.03)
        # print(state.pose.position.y)

    def get_human_states(self):
        obstacle = []
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        # print(model.name)
        human_states = []
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "dynamic_obstacles"
        marker.type = Marker.CYLINDER  # 圆柱体
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration()
        obstacle_positions = []

        for i in range(self.human_num):
            # for j in range(len(model.name)):
            if model.name[i] == 'human_' + str(i + 1):
                model_state = self.get_model_state('human_' + str(i + 1), 'world')

                px = model_state.pose.position.x
                py = model_state.pose.position.y
                vx = model_state.twist.linear.x
                vy = model_state.twist.linear.y
                # print('human_' + str(i + 1), px, py, vx, vy)
                obstacle_positions.append([px, py, 0])
                human_states.append(ObservableState(px, py, vx, vy, 0.4))
            # for position in obstacle_positions:
        for i in range(self.human_num):
            marker.id = i
            marker.pose.position = Point(*obstacle_positions[i])
            self.marker_pub.publish(marker)

        return human_states

    def get_robot_states(self):
        # self.get_robot_velocity()
        # speed_msg = rospy.wait_for_message('/cmd_vel', Twist)
        # print(speed_msg)
        model_state = self.get_model_state('turtlebot3_waffle', 'world')
        # self.linear_velocity, self.angular_velocity = self.return_velocity()
        # self.linear_velocity, self.angular_velocity = model_state.twist.linear.x, model_state.twist.angular.z
        # position = self.get_odom()
        # print(position)
        # px = position.x
        # py = position.y
        # vx = self.linear_velocity * math.cos(self.angular_velocity)
        # vy = self.linear_velocity * math.sin(self.angular_velocity)
        px = model_state.pose.position.x
        py = model_state.pose.position.y
        vx = model_state.twist.linear.x
        vy = model_state.twist.linear.y
        self.theta = pi / 2
        # print(px, py, vx, vy)
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
        for j in range(1):
            if model.name[i] == 'human_' + str(i + 1):
                # obstacle.append(ModelState())
                # print(model.name[j])
                obstacle[i].model_name = model.name[i]
                obstacle[i].pose = model.pose[i]
                # obstacle[i].twist = Twist()
                # print('human_' + str(i + 1), action.vx, action.vy)
                obstacle[i].twist.linear.x = action.vx
                obstacle[i].twist.linear.y = action.vy
                self.humans[i].vx = obstacle[i].twist.linear.x
                self.humans[i].vy = obstacle[i].twist.linear.y
                self.humans[i].px = obstacle[i].pose.position.x
                self.humans[i].py = obstacle[i].pose.position.y
                self.pub_model.publish(obstacle[i])

                # state_human = self.get_model_state('human_' + str(i + 1), "world")
                # print('human_' + str(i + 1), state_human.twist.linear)
                rospy.sleep(0.03)
                r.sleep()
            # print('human_sate:' + str(i + 1), state_human)

    def reset(self, phase='test', test_case=None):
        """
            Set px, py, gx, gy, vx, vy, theta for robot and humans
            :return:
        """
        # print 'env reset in crow_sim'
        self.init_ros()
        self.reset_proxy()
        # print('set')
        # self.get_robot_pose()
        # self.marker.id = 0
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

        r = rospy.Rate(5000)
        # get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for j in range(1):
            for i in range(self.human_num):
                if model.name[i] == 'human_' + str(i + 1):
                    # print(model.name[i])
                    obstacle.model_name = model.name[i]
                    obstacle.pose = model.pose[i]
                    obstacle.pose.position.x = ob[i].px
                    obstacle.pose.position.y = ob[i].py
                    self.pub_model.publish(obstacle)
                    # time.sleep(0.04)
            r.sleep()

        self.robot_path.poses = []
        self.info_robot_path_pub.publish(self.robot_path)

        self.goal_info_marker.text = "Goal"
        self.goal_info_marker.pose.position.x = self.robot.gx
        self.goal_info_marker.pose.position.y = self.robot.gy
        self.info_goal_marker_pub.publish(self.goal_info_marker)
        return ob

    def step(self, action, update=True):
        self.info_goal_marker_pub.publish(self.goal_info_marker)
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

        # v_linear = sqrt(action.vx ** 2 + action.vy ** 2)
        # v_angular = atan2(action.vy, action.vx)
        self.currentTime = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

        self.delta_t = self.currentTime - self.lastTime

        self.step_counter = self.step_counter + 1
        self.lastTime = self.currentTime
        self.global_time = self.global_time + self.delta_t
        # self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + v_linear, -self.robot.v_pref, self.robot.v_pref)
        # self.desiredVelocity[1] = v_angular / self.fixed_time_interval

        # v_smooth, w_smooth = self.smooth(self.desiredVelocity[0], self.desiredVelocity[1])
        # self.last_v = v_smooth
        # print(action)
        realAction = Twist()
        realAction.linear.x = action.v
        # realAction.linear.y = action.vy
        realAction.angular.z = action.r

        dmin = float('inf')
        collision = False
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        position = self.get_odom()
        px, py = position.x, position.y
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
                    pow(x - px, 2) + pow(y - py, 2)) - 0.8
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
        self.currentTime_else = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

        # rospy.sleep(0.04)  # act as frame skip

        # time.sleep(0.1)
        # rospy.wait_for_service("/gazebo/unpause_physics")
        position = self.get_odom()
        # get_v = self.get_model_state("turtlebot3_waffle", "world")
        # print(get_v)
        goal_dist = sqrt(pow(self.gx - position.x, 2) + pow(self.gy - position.y, 2))
        # print(goal_dist)
        reaching_goal = goal_dist <= self.robot.radius + 0.2
        if self.global_time >= self.time_limit:
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
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor
            done = False
            info = Danger(dmin)
        else:
            potential_cur = goal_dist
            reward = 2 * (self.potential - potential_cur)
            self.potential = abs(potential_cur)
            done = False
            info = Nothing()

        self.cmd_vel.publish(realAction)
        for i, human_action in enumerate(human_actions):
            # print(i)
            obstacle.append(ModelState())
            self.human_step(i, obstacle, human_action)
        # reward = None
        ob = self.get_human_states()

        self.get_robot_pose()
        return ob, reward, done, info
