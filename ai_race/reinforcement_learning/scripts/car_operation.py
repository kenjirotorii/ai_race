#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import math
import subprocess

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64
from nav_msgs.msg import Odometry
import tf

import numpy as np
from PIL import Image as IMG
import cv2
from cv_bridge import CvBridge

import torch
import torchvision

from utils.utility import state_transition, distance_from_centerline
from agents.deepQlearning import Agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 5

class CarBot:

    def __init__(self):

        # node name
        rospy.init_node('car_bot', anonymous=True)

        # Publisher
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Subscriber
        self.pose_sub = rospy.Subscriber('/tracker', Odometry, self.callback_odom)

        # Initial twist
        self.twist_pub.publish(Twist())

        # odom
        self.odom_x = 1.6
        self.odom_y = 0.0
        self.odom_theta = 1.57

        self.bridge = CvBridge()
        self.images = []
        self.actions = []
        self.rewards = []

        self.course_out = False

        # agent
        self.agent = Agent(num_actions=NUM_ACTIONS, mem_capacity=1000, batch_size=32, lr=0.0005, gamma=0.95, debug=True)

    def callback_odom(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        q = (qx, qy, qz, qw)
        e = tf.transformations.euler_from_quaternion(q)
        self.odom_theta = e[2]

    def set_throttle_steering(self, data):

        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        image = IMG.fromarray(image)
        image = torchvision.transforms.ToTensor()(image)    # (3, 240, 320)
        # use only lower parts of images
        image = image[:, 120:, :]                           # (3, 120, 320)
        image = torch.unsqueeze(image, 0)                   # (1, 3, 120, 320)

        self.images.append(image)

        # get action from agent
        image = image.to(DEVICE)
        action = self.agent.get_action(image, self.episode).to('cpu')
        self.actions.append(action)

        angular_z = float(float(action[0])-((NUM_ACTIONS-1)/2))/((NUM_ACTIONS-1)/2)

        current_pose = np.array([self.odom_x, self.odom_y, self.odom_theta])
        next_pose = state_transition(pose=current_pose, omega=angular_z, vel=1.6, dt=0.1)

        reward = self.get_reward(next_pose)
        self.rewards.append(reward)

        # update twist
        twist = Twist()
        twist.linear.x = 1.6
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self.twist_pub.publish(twist)

        self.step += 1

        rospy.loginfo('epi=%d, step=%d, action=%d, reward=%4.2f' % (self.episode, self.step, action, reward))

    def get_reward(self, pose):
        dist_from_center = distance_from_centerline(pose)

        if dist_from_center < 0.1:
            return 1.0
        elif dist_from_center < 0.2:
            return 1.0
        elif dist_from_center < 0.3:
            return 1.0
        elif dist_from_center < 0.4:
            return 1.0
        elif dist_from_center < 0.45:
            return 0.0
        elif dist_from_center < 0.6:
            return -1.0
        else:
            self.course_out = True
            rospy.loginfo('Course Out !!')
            return -1.0

    def stop(self):
        rospy.loginfo('***** EPISODE #%d *****' % (self.episode))
        rospy.loginfo('total step:%d' % (self.step))
        rospy.loginfo('***********************')
        # stop car
        self.twist_pub.publish(Twist())
        # unregister image subscription
        self.image_sub.unregister()

        # push data to agent's memory
        for i in range(self.step):
            img = self.images[i].to(DEVICE)
            if i != self.step - 1:
                next_img = self.images[i + 1].to(DEVICE)
            else:
                next_img = None

            act = self.actions[i].to(DEVICE)
            rwd = torch.LongTensor([self.rewards[i]]).to(DEVICE)

            self.agent.memorize(img, act, next_img, rwd)

    def agent_training(self, n_epoch):
        # Experience ReplayでQ関数を更新する
        for epoch in range(n_epoch):
            self.agent.update_q_function()

    def update_target_q(self):
        self.agent.update_target_q_function()

    def restart(self):

        self.step = 0
        self.course_out = False
        
        self.images = []
        self.actions = []
        self.rewards = []

        # Initial twist
        self.twist_pub.publish(Twist())

        # initialize judge and car pose
        subprocess.call('bash ~/catkin_ws/src/ai_race/ai_race/reinforcement_learning/scripts/utils/reset.sh', shell=True)

        time.sleep(1)
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

    def run(self):
        
        self.episode = 0
        self.step = 0
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

        r = rospy.Rate(30)

        while not rospy.is_shutdown():

            if self.course_out:
                
                self.stop()
                self.agent_training(n_epoch=20)
                self.agent.save_model('../model_weight/dqn_20210102.pth')

                self.episode += 1
                # update target q-function every 2 episodes
                if self.episode % 2 == 0:
                    self.update_target_q()

                self.restart()

            r.sleep()


if __name__ == "__main__":
    
    car_bot = CarBot()

    car_bot.run()
