import threading

import numpy as np
import rospy
import tf
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist, PoseStamped
import gazebo_msgs
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates, ModelState
from nav_msgs.msg import Odometry
from numpy.linalg import norm


class Moving:
    def __init__(self):
        # 初始化ROS节点
        # rospy.init_node('gazebo_listener', anonymous=True)
        rospy.Subscriber('/gazebo/velocity', Twist, self.velocity_callback)
        # self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=0)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.cmd_vel.publish(Twist())
        self.speed = None
        # self.moving()
        # Robot Path
        self.use_amcl_pose = True
        self.info_robot_path_pub = rospy.Publisher('/robot_path_info', Path, queue_size=20)
        self.robot_path = Path()
        self.robot_path.header.stamp = rospy.Time.now()
        self.robot_path.header.frame_id = 'map'
        # self.lock = threading.Lock()
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # self.odom_sub = rospy.Subscriber("/current_pose", PoseStamped, self.odom_callback, queue_size=3)
        # self.planner_thread = threading.Thread(target=self.moving(), name='PlanningThread')
        self.rate = rospy.Rate(100)
        # self.moving.start()

    def velocity_callback(self, data):
        self.speed = data.linear.x

    def get_speed(self):
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state('turtlebot3_burger', 'world')
        # print(model_state.twist.linear.x, model_state.twist.angular.z)

    def odom_callback(self, odom):
        # def odom_update(self, odom):
        print(odom)
        # self.lock.acquire()
        # try:
        self.robot_path.poses.append(odom)
        self.info_robot_path_pub.publish(self.robot_path)
        self.rate.sleep()
        # finally:
        #     se()

    def moving(self):
        r = rospy.Rate(500)
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        while not rospy.is_shutdown():
            action = Twist()
            action.linear.x = 0.3
            action.angular.z = 0.9
            self.cmd_vel.publish(action)
            rospy.sleep(0.03)
            pose = self.get_model_state("turtlebot3_waffle", "world")
            odom = pose
            self.robot_path.poses.append(odom)
            self.info_robot_path_pub.publish(self.robot_path)
            rospy.sleep(0.03)



def main():
    rospy.init_node('gazebo_listener', anonymous=True)
    moving = Moving()
    # speed = moving.get_speed()
    # while not rospy.is_shutdown():
    # print(speed)
    # if speed is not None:
    #     print("Current speed: {}".format(speed))
    # rospy.sleep(0.5)
    moving.moving()
    # print(state)
    rospy.spin()


if __name__ == '__main__':
    main()
