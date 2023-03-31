import numpy as np
import tf
import copy

class Coordinate_transformation(object):
    def __init__(self):
        super(Coordinate_transformation, self).__init__()

    def rot(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), np.sin(roll)],
                    [0, -np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                    [0, 1, 0],
                    [np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        R = Rz.dot(Ry).dot(Rx)

        return R

    def axis_normalize(self, quaternion):
        euler = list(tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w)))
        return list(tf.transformations.quaternion_from_euler(euler[2], euler[0], euler[1]))

    def transform(self, local_pose, offset):
        pose = local_pose
        R = self.rot(offset[3:6])
        position = np.array([local_pose.position.x, local_pose.position.y, local_pose.position.z]) + np.array(offset[0:3])
        position = np.dot(R.T, np.array([local_pose.position.x, local_pose.position.y, local_pose.position.z]))
        position[1] *= -1
        position = position + np.array(offset[0:3])

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2] * -1

        ori = self.axis_normalize(pose.orientation)
        pose.orientation.x = ori[0]
        pose.orientation.y = ori[1]
        pose.orientation.z = ori[2]
        pose.orientation.w = ori[3]
        

        return pose
        
