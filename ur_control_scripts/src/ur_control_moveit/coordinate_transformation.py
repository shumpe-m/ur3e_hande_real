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

        R = Rz.T.dot(Ry.T).dot(Rx.T) # .T = transform right handed axis to left handed axis

        return R

    def transform_rightHanded_to_leftHanded(self, orientation):
        ori = copy.deepcopy(orientation)
        ori.x = -orientation.x
        ori.y = orientation.z
        ori.z = orientation.y
        ori.w = -orientation.w
        return ori

    def transform(self, local_pose, offset):
        # print(local_pose.position)
        pose = local_pose
        R = self.rot(offset[3:6])
        position = np.array([local_pose.position.x, local_pose.position.y, local_pose.position.z])
        position = np.dot(R.T, position)
        print(position)
        position = position + np.array(offset[0:3])

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        ori = self.transform_rightHanded_to_leftHanded(pose.orientation)
        pose.orientation = ori

        return pose
        
