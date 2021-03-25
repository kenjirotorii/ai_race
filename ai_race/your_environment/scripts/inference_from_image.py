#!/usr/bin/env python

import os
import sys
import time
import argparse

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64

import numpy as np
from PIL import Image as IMG
import cv2
from cv_bridge import CvBridge

import torch
import torchvision

from networks.autoencoder import ControlNet

DISCRETIZATION = 3
CWD_PATH = os.path.dirname(os.path.abspath(__file__))

model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_inference():
    global model
    global device

    model = ControlNet(80, 160, 256, DISCRETIZATION, args.variational)

    model.eval()

    if args.trt_module:
        from torch2trt import TRTModule

        if args.trt_conversion:
            model.load_state_dict(torch.load(args.pretrained_model))
            model = model.cuda()
            x = torch.ones((1, 3, 80, 160)).cuda()
            from torch2trt import torch2trt
            model_trt = torch2trt(model, [x], max_batch_size=100, fp16_mode=True)
            torch.save(model_trt.state_dict(), args.trt_model)
            exit()
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(args.trt_model))
        model = model_trt.to(device)

    else:
        model.load_state_dict(torch.load(args.pretrained_model))
        model = model.to(device)

i = 0
pre = time.time()
now = time.time()
bridge = CvBridge()
twist = Twist()

def set_throttle_steer(data):

    global i
    global pre
    global now
    global bridge
    global twist
    global model
    global device

    i=i+1
    if i == 100 :
        pre = now
        now = time.time()
        i = 0
        print ("average_time:{0}".format((now - pre)/100) + "[sec]")
    start = time.time()
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = image[120:, :]
    image = cv2.resize(image, (160, 80))
    image = IMG.fromarray(image)
    image = torchvision.transforms.ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    image = image.to(device)
    outputs = model(image)

    outputs_np = outputs.to('cpu').detach().numpy().copy()
    print(outputs_np)
    output = np.argmax(outputs_np, axis=1)
    print(output)

    angular_z = float(output)

    twist.linear.x = 1.6
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = angular_z
    twist_pub.publish(twist)
    end = time.time()
    print ("time_each:{0:.3f}".format((end - start)) + "[sec]")

def inference_from_image():
    global twist_pub
    rospy.init_node('inference_from_image', anonymous=True)
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber("/front_camera/image_raw", Image, set_throttle_steer)
    r = rospy.Rate(10)
    rospy.spin()

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Autonomous with inference")

    arg_parser.add_argument("--trt_conversion", action='store_true')
    arg_parser.add_argument("--trt_module", action='store_true')
    arg_parser.add_argument("--pretrained_model", type=str, default=CWD_PATH+'/models/control_z256_ckpt_25.pth')
    arg_parser.add_argument("--trt_model", type=str, default=CWD_PATH+'/trt_models/dqn_20210113_trt.pth' )
    arg_parser.add_argument("--variational", action='store_true')

    args = arg_parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    init_inference()
    try:
        inference_from_image()
    except rospy.ROSInterruptException:
        pass

