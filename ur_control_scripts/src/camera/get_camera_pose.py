import geometry_msgs.msg
import tf2_geometry_msgs
import rospy
import tf2_ros
from utils.transformations import Transformations

import numpy as np
import ctypes
import struct
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs import point_cloud2

class GetCameraPose():
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tool0_to_camera = [-0.0371687, -0.0789837, -0.0158126, 0.492351, -0.512039, 0.490376, 0.504914]

    def get_link_pose(self, link_name='camera_color_frame'):
        # Translate from map coordinate to arbitrary coordinate of robot.
        link_pose = geometry_msgs.msg.PoseStamped()
        link_pose.header.frame_id = link_name
        link_pose.header.stamp = rospy.Time(0)
        link_pose.pose.orientation.w = 1.0

        try:
            # Get transform at current time
            base_to_link_pose = self.tfBuffer.transform(link_pose, 'base_link')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return None

        return base_to_link_pose


class GetPointCloud():
    def __init__(self):
        # ros message
        self.sub_vector = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.callbackVector)
        self.pointcloud = PointCloud2()

    def callbackVector(self, ptcloud_data):
        self.pointcloud = ptcloud_data

    def get_pose(self):
        while self.pointcloud.header.frame_id == '':
            pass
        gen = point_cloud2.read_points(self.pointcloud, skip_nans=True)
        int_data = list(gen)

        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        for idx in int_data:
            test = idx[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            xyz = np.append(xyz,[[idx[0],idx[1],idx[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)

        return xyz, rgb

class GetDepthImg():
    def __init__(self):
        # ros message
        self.sub_vector = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callbackVector)
        self.depth = Image()

    def callbackVector(self, msg):
        self.depth = msg

    def get_depth(self):
        while self.depth.header.frame_id == '':
            pass

        return self.depth

class GetColorImg():
    def __init__(self):
        # ros message
        self.sub_vector = rospy.Subscriber("/camera/color/image_raw", Image, self.callbackVector)
        self.rgb = Image()

    def callbackVector(self, msg):
        self.rgb = msg

    def get_rgb(self):
        while self.rgb.header.frame_id == '':
            pass

        return self.rgb

class GetDepthInfo():
    def __init__(self):
        # ros message
        self.sub_vector = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.callbackVector)
        self.depthInfo = CameraInfo()

    def callbackVector(self, msg):
        self.depthInfo = msg

    def get_depth_info(self):
        while self.depthInfo.header.frame_id == '':
            pass
        depth_info = np.array(list(self.depthInfo.K))
        depth_info = np.reshape(depth_info, [3, 3])
        print(depth_info)

        return depth_info