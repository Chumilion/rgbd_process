#!/usr/bin/env python

import rospy
import copy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import cv2
import cv_bridge

class RGBDCalibrator:
    def __init__(self, shift_coef): #crop = None if no crop

        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('calibrator', anonymous=True)
        self.shift_coef = shift_coef

        self.bridge = cv_bridge.CvBridge()
        self.sub_color_img = rospy.Subscriber("camera/color/image_raw",
                                              Image, self.manipulate_color_img)
        self.sub_color_info = rospy.Subscriber("camera/color/camera_info",
                                               CameraInfo, self.manipulate_color_topic)
        self.sub_depth_img = rospy.Subscriber("camera/depth/image_raw",
                                              Image, self.manipulate_depth_img)
        self.pub_color_img = rospy.Publisher("calibrate_camera/color/image_raw",
                                             Image, queue_size=10)
	self.pub_color_info = rospy.Publisher("calibrate_camera/color/camera_info",
                                              CameraInfo, queue_size=10)
        self.pub_depth_img = rospy.Publisher("calibrate_camera/depth/image_raw",
                                             Image, queue_size=10)
	self.camera_info_buffer = None
        rospy.spin()

    def calibrated_color_img(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
        run_dim = self.in_dim
        if self.border:
            cv_image = cv2.copyMakeBorder(cv_image, top=self.border,
                                                    bottom=self.border,
                                                    left=self.border,
                                                    right=self.border,
                                                    borderType=cv2.BORDER_CONSTANT,
                                                    value=[0, 0, 0])
        if self.crop:
            cv_image = cv_image[self.crop[2]:self.crop[3], self.crop[0]:self.crop[1]]
            run_dim = (self.crop[1] - self.crop[0], self.crop[3] - self.crop[2])
        cv_image = cv2.resize(cv_image, (0, 0),
                              fx=float(self.out_dim[0])/run_dim[0],
                              fy=float(self.out_dim[1])/run_dim[1])

        cv_image = cv2.Canny(cv_image, 100, 200)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        manipulated = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
        manipulated.header = copy.deepcopy(data.header)
        manipulated.header.stamp = rospy.get_rostime()
        #manipulated.header.frame_id = "camera_depth_optical_frame"
        self.camera_info_buffer.header = copy.deepcopy(data.header)
        self.pub_color_info.publish(self.camera_info_buffer)
        self.pub_color_img.publish(manipulated)

    def manipulate_depth_img(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv_image = cv2.convertScaleAbs(cv_image)

        cv_image = cv2.Canny(cv_image, 100, 200)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        manipulated = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
        self.pub_depth_img.publish(manipulated)

    def manipulate_color_topic(self, data):
        new_data = copy.deepcopy(data)
        new_data.height = self.out_dim[1]
        new_data.width = self.out_dim[0]
        new_K = list(data.K)
        new_P = list(data.P)
        shift = (-self.crop[0], -self.crop[2]) if self.crop else (0, 0)
        scale = (float(self.out_dim[0])/(self.crop[1] - self.crop[0]),
                 float(self.out_dim[1])/(self.crop[3] - self.crop[2])) if self.crop else \
                (float(self.out_dim[0] + 2*self.border)/self.in_dim[0],
                 float(self.out_dim[1] + 2*self.border)/self.in_dim[1])
        new_K[0:3] = [(elem + shift[0])*scale[0] for elem in data.K[0:3]]
        new_K[3:6] = [(elem + shift[1])*scale[1] for elem in data.K[3:6]]
        new_P[0:4] = [(elem + shift[0])*scale[0] for elem in data.P[0:4]]
                new_P[4:8] = [(elem + shift[1])*scale[1] for elem in data.P[4:8]]

        new_data.K = tuple(new_K)
        new_data.P = tuple(new_P)
        self.camera_info_buffer = new_data

if __name__ == '__main__':
    #CameraManipulator((640, 480), 75, (5, 645, 75, 555), (480, 360))
    CameraManipulator((640, 480), 0, None, (480, 360))

                                                                                  101,0-1       Bot


