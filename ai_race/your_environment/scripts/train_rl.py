#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import argparse

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

from rl.utils import state_transition, distance_from_centerline, distance_from_inline
from rl.ddqn import Agent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME_PATH = os.environ['HOME']

class CarBot:

    def __init__(self, args):
        
        self.save_model_path = args.model_path + args.model_name + '.pth'
        self.online = args.online_learning
        self.n_epoch = args.n_epoch
        self.target_update_interval = args.target_update_interval

        # node name
        rospy.init_node('car_bot', anonymous=True)

        # Publisher
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Subscriber
        self.pose_sub = rospy.Subscriber('wheel_robot_tracker', Odometry, self.callback_odom)

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
        self.episode = args.start_episode
        self.complete = 0

        # agent
        self.agent = Agent(img_size=(80, 160), num_actions=args.num_actions, num_latent=args.num_z,
                            mem_capacity=args.memory_cap, batch_size=args.batch_size, lr=args.lr, 
                            gamma=args.gamma, debug=True, device=DEVICE, 
                            pretrained_model=args.pretrained_model, variational=args.variational)

        if args.trained_model:
            self.agent.load_model(model_path=args.trained_model)

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
        image = image[120:, :]
        image = cv2.resize(image, (160, 80))
        image = IMG.fromarray(image)
        image = torchvision.transforms.ToTensor()(image)    # (3, 80, 160)
        image = torch.unsqueeze(image, 0)                   # (1, 3, 80, 160)

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
        
        rospy.sleep(0.01)

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
            return -1.0
        elif dist_from_inline < 0.3:
            return 1.0
        elif dist_from_inline < 0.6:
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
        subprocess.call('bash ~/catkin_ws/src/ai_race/ai_race/your_environment/scripts/rl/reset.sh', shell=True)

        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

    def run(self):
        
        self.step = 0
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.set_throttle_steering)

        r = rospy.Rate(30)

        while not rospy.is_shutdown():
            
            if self.course_out or self.step > 2400:
                
                self.stop()

                if self.step > 2400:
                    self.complete += 1
                    complete_path = self.save_model_path.split('.')[0] + '_complete_{}.pth'.format(self.episode)
                    self.agent.save_model(model_path=complete_path)
                    
                    if self.complete >= 3:
                        break
                else:
                    self.complete = 0

                print("agent model save to {}".format(self.save_model_path))
                self.agent.save_model(model_path=self.save_model_path)

                if not self.online:
                    self.agent_training(self.n_epoch)

                self.episode += 1
                
                if self.episode % self.target_update_interval == 0:
                    self.update_target_q()

                self.restart()

            r.sleep()
        

def parse_args():

    arg_parser = argparse.ArgumentParser(description="DDQN Training")
    
    arg_parser.add_argument("--variational", action='store_true')
    arg_parser.add_argument("--online_learning", action='store_true')
    arg_parser.add_argument("--model_path", type=str, default=HOME_PATH+'/catkin_ws/src/ai_race/ai_race/your_environment/models/')
    arg_parser.add_argument("--model_name", type=str, default='ddqn_model')
    arg_parser.add_argument("--pretrained_model", type=str, default=HOME_PATH+'/catkin_ws/src/ai_race/ai_race/your_environment/models/vae_mse_z256_ckpt_25.pth')
    arg_parser.add_argument("--trained_model", type=str, default=None)
    arg_parser.add_argument('--num_actions', default=2, type=int, help='The number of actions')
    arg_parser.add_argument('--num_z', default=256, type=int, help='The number of latent variables')
    arg_parser.add_argument('--memory_cap', default=1000, type=int, help='Meomry capacity of replay buffer')
    arg_parser.add_argument('--n_epoch', default=25, type=int, help='The number of epoch')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    arg_parser.add_argument('--gamma', default=0.99, type=float, help='Discount rate')
    arg_parser.add_argument('--target_update_interval', default=5, type=int, help='target update interval')
    arg_parser.add_argument('--start_episode', default=0, type=int)

    args = arg_parser.parse_args()

    # Make directory.
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Validate paths.
    assert os.path.exists(args.model_path)
    assert os.path.exists(args.pretrained_model)

    return args


def main():
    # Parse arguments.
    args = parse_args()
    
    # rl training.
    car_bot = CarBot(args=args)
    car_bot.run()


if __name__ == "__main__":
    main()
    print("finished successfully.")
    os._exit(0)
