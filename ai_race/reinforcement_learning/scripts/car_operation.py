#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
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

from utils.utility import state_transition, distance_from_centerline, distance_from_inline
from agents.deepQlearning import Agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarBot:

    def __init__(self, save_model_path=None, pretrained=False, load_model_path=None, online=True):
        
        if save_model_path:
            self.save_model_path = save_model_path
        else:
            self.save_model_path = '../model_weight/test.pth'

        self.online = online

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
        self.episode = 0
        self.complete = 0

        # agent
        self.agent = Agent(img_size=(120, 320), num_actions=NUM_ACTIONS, mem_capacity=CAPACITY, batch_size=BATCH_SIZE, lr=LR, gamma=GAMMA, debug=True)

        if pretrained:
            self.agent.load_model(model_path=load_model_path)

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

        if self.episode < 30:
            eps = 0.2
        elif self.episode < 50:
            eps = 0.1
        else:
            eps = 0.0

        action = self.agent.get_action(state=image, epsilon=eps).to('cpu')
        self.actions.append(action)

        angular_z = float(action[0])
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

        if self.online:
            self.agent_training(n_epoch=1)
        self.step += 1

        rospy.loginfo('epi=%d, step=%d, action=%d, reward=%4.2f' % (self.episode, self.step, action, reward))

    def get_reward(self, pose):
        
        dist_from_inline = distance_from_inline(pose)
        
        if dist_from_inline < -0.10:
            self.course_out = True
            rospy.loginfo('Course Out !!')
            return -1.0
        elif dist_from_inline < 0:
            return -0.5
        elif dist_from_inline < 0.2: # 0.3
            return 1.0
        elif dist_from_inline < 0.45: # 0.5
            return 0.0
        elif dist_from_inline < 0.9:
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
        print("agent training n_epoch:{}".format(n_epoch))
        for epoch in range(n_epoch):
            self.agent.update_q_function()

    def agent_model_save(self):
        print("agent model save to {}".format(self.save_model_path))
        self.agent.save_model(model_path=self.save_model_path)

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

        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

    def run(self):
        
        self.step = 0
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

        r = rospy.Rate(30)

        while not rospy.is_shutdown():
            
            if self.course_out or self.step > 2500:
                
                self.stop()

                if self.step > 2500:
                    self.complete += 1
                    if self.complete >= 3:
                        break
                    self.agent_model_save()
                else:
                    self.complete = 0
                
                if not self.online:
                    self.agent_training(n_epoch=25)

                self.episode += 1
                # update target q-function every 2 episodes
                if self.episode % TARGET_UPDATE == 0:
                    self.update_target_q()

                self.restart()

            r.sleep()
        

if __name__ == "__main__":
    
    SAVE_MODEL_PATH = '../model_weight/dqn_20210114.pth'
    LOAD_MODEL_PATH = '../model_weight/dqn_20210108.pth'
    
    # parameters
    NUM_ACTIONS = 2
    CAPACITY = 1000
    BATCH_SIZE = 32
    LR = 0.0005
    GAMMA = 0.995
    TARGET_UPDATE = 3

    car_bot = CarBot(save_model_path=SAVE_MODEL_PATH, pretrained=False, load_model_path=LOAD_MODEL_PATH, online=False)
    car_bot.run()
