import subprocess
import tf_conversions
import sys
import math
import time

def put_vehicle_rpy( x, y, z, roll, pitch, yaw ) :
    quot = tf_conversions.transformations.quaternion_from_euler( roll, pitch, yaw )

    cmd_str = "rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState '{ model_name: wheel_robot, pose: { position: { x: " + str( x ) + ", y: " + str( y ) + ", z: " + str( z ) + " }, orientation: {x: " + str( quot[0] ) + ", y: " + str( quot[1] ) + ", z: " + str( quot[2] ) + ", w: " + str( quot[3] ) + " } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, }'"

    print cmd_str

    print subprocess.call( cmd_str, shell=True )
    #print subprocess.call( "ls" )

import rospy
from gazebo_msgs.msg import ModelState
class PutVehicle :
    def __init__( self ) :
        self.modelState = ModelState()
        self.publisherVehiclePos = rospy.Publisher( '/gazebo/set_model_state', ModelState, queue_size=1 )

    def send( self ) :
        #print( self.modelState )
        self.publisherVehiclePos.publish( self.modelState )

    def set( self, x, y, z, roll, pitch, yaw ) :
        quot = tf_conversions.transformations.quaternion_from_euler( roll, pitch, yaw )
        self.modelState.model_name = 'wheel_robot'
        self.modelState.pose.position.x = x
        self.modelState.pose.position.y = y
        self.modelState.pose.position.z = z
        self.modelState.pose.orientation.x = quot[0]
        self.modelState.pose.orientation.y = quot[1]
        self.modelState.pose.orientation.z = quot[2]
        self.modelState.pose.orientation.w = quot[3]
        self.modelState.twist.linear.x = 0
        self.modelState.twist.linear.y = 0
        self.modelState.twist.linear.z = 0
        self.modelState.twist.angular.x = 0
        self.modelState.twist.angular.y = 0
        self.modelState.twist.angular.z = 0

class AutoRun :
    def __init__( self ) :
        self.step = 0.1
        self.north_x = 0.6
        self.south_x = -0.6
        self.west_y = 1.2
        self.east_y = -1.2

        self.position = [1.7, 0]
        #self.position = [1.3, 0]
        self.heading = math.pi / 2

    def move( self ) :
        area = self.area()
        if area == 'area1' :
            self.position[1] += self.step
            self.heading = math.pi / 2
        elif area == 'area3' :
            self.position[0] -= self.step
            self.heading = math.pi
        elif area == 'area5' :
            self.position[1] -= self.step
            self.heading = math.pi * 3 / 2
        elif area == 'area7' :
            self.position[0] += self.step
            self.heading = 0
        elif area == 'area2' :
            self.corner( self.north_x, self.west_y )
        elif area == 'area4' :
            self.corner( self.south_x, self.west_y )
        elif area == 'area6' :
            self.corner( self.south_x, self.east_y )
        elif area == 'area8' :
            self.corner( self.north_x, self.east_y )
        else :
            print( 'area error' )

    def area( self ) :
        if self.position[0] > self.north_x :
            if self.position[1] > self.west_y :
                ret = 'area2'
            elif self.position[1] < self.east_y :
                ret = 'area8'
            else :
                ret = 'area1'
        elif self.position[0] < self.south_x :
            if self.position[1] > self.west_y :
                ret = 'area4'
            elif self.position[1] < self.east_y :
                ret = 'area6'
            else :
                ret = 'area5'
        else :
            if self.position[1] >= 0 :
                ret = 'area3'
            else :
                ret = 'area7'
        #print( ret )
        return ret

    def calcTheta( self, x, y ) :
        radius = math.sqrt( x ** 2 + y ** 2 )
        theta = self.step / radius
        #print( theta )
        return theta

    def rotate( self, theta, x, y ) :
        retX = x * math.cos( theta ) - y * math.sin( theta )
        retY = y * math.cos( theta ) + x * math.sin( theta )
        #print( x, y, retX, retY )
        return retX, retY

    def corner( self, originX, originY ) :
        oftX = self.position[0] - originX
        oftY = self.position[1] - originY
        theta = self.calcTheta( oftX, oftY )
        rotX, rotY = self.rotate( theta, oftX, oftY )
        self.position[0] = rotX + originX
        self.position[1] = rotY + originY
        self.heading += theta


from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import datetime
import os

class PictureSave :
    def __init__( self ) :
        self.savePath = '/home/hiro/ai_race/Images_from_rosbag/_' + self.now() + '/images'
        #self.savePath = '/home/hiro/ai_race/Images_from_rosbag'

        self.image = None
        self.flameNum = 0

        #rospy.init_node('picture_save_node', anonymous=True)
        self.subscriberImage = rospy.Subscriber( 'front_camera/image_raw', Image, self.callback_image_topic )

        os.makedirs( self.savePath )

    def callback_image_topic( self, image ) :
        self.image = image
        #print( 'image_topic' )

    def save( self, postfix='' ) :
        filename = str( self.flameNum ).zfill(6) + postfix + '.jpg'
        #print( filename )
        if self.image is None :
            print( 'dose not receive image topic' )
            return
        image = CvBridge().imgmsg_to_cv2(self.image, "bgr8")
        print( self.savePath + '/' + filename )
        cv2.imwrite( self.savePath + '/' + filename, image )
        self.flameNum += 1

    def now( self ) :
        dat = datetime.datetime.now()
        timestamp = str(dat.year).zfill(4)+'-'+str(dat.month).zfill(2)+'-'+str(dat.day).zfill(2)+'-'+str(dat.hour).zfill(2)+'-'+str(dat.minute).zfill(2)+'-'+str(dat.second).zfill(2)
        return timestamp

def set_pos_and_save_pic() :
    rospy.init_node('set_pos_and_save_pic_node', anonymous=True)
    Auto = AutoRun()
    Picture = PictureSave()
    PV = PutVehicle()
    time.sleep( 3 )
    for i in [ 1+float(i)/10 for i in range(16) ] :
        print( i )
        Auto.position = [ i, 0 ]
        Auto.hedding = math.pi / 2
        prevY = 0
        while True :
            prevY = Auto.position[1]
            Auto.move()
            print( Auto.position[0], Auto.position[1], Auto.heading )
            if prevY < 0 and Auto.position[1] >= 0 :
                break
            #put_vehicle_rpy( Auto.position[0], Auto.position[1], 0, 0, 0, Auto.heading )
            HEADING_OFFSET = 0 # math.pi/180*20
            PV.set( Auto.position[0], Auto.position[1], 0.0, 0.0, 0.0, Auto.heading + HEADING_OFFSET )
            PV.send()
            time.sleep( 0.1 )
            postfix = '_' + str( int( Auto.position[0] * 1000 ) ).zfill( 5 ) + '_' + str( int( Auto.position[1] * 1000 ) ).zfill( 5 ) + '_' + str( int( Auto.heading * 1000 ) ).zfill( 4 )
            #print( postfix )
            Picture.save( postfix )

def test_PutVehicle() :
    rospy.init_node('set_pos_and_save_pic_node', anonymous=True)
    ros = rospy.Rate( 2 )
    PV = PutVehicle()
    for i in range( 100 ) :
        PV.set( 1.7, float(i)/10, 0.0, 0, 0, 0 )
        PV.send()
        ros.sleep()


if __name__ == '__main__' :
    argv = sys.argv
    argc = len( argv )

    cmd = argv[1]

    if cmd == 'put' :

        x = float( argv[2] )
        y = float( argv[3] )
        z = float( argv[4] )
        roll = float( argv[5] )
        pitch = float( argv[6] )
        yaw = float( argv[7] )

        put_vehicle_rpy( x, y, z, roll, pitch, yaw )

    elif cmd == 'auto' :
        Auto = AutoRun()
        for i in range( 1000 ) :
            Auto.move()
            print( Auto.position[0], Auto.position[1], Auto.heading )
            put_vehicle_rpy( Auto.position[0], Auto.position[1], 0, 0, 0, Auto.heading )

    elif cmd == 'auto_pic' :
        set_pos_and_save_pic()

    elif cmd == 'test_put_pub' :
        test_PutVehicle()
