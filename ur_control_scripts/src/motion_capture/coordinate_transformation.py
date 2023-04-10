import numpy as np
import tf
import copy

class Coordinate_transformation(object):
    def __init__(self):
        super(Coordinate_transformation, self).__init__()

    def rot(self, rpy):
        """
        This function calculates a rotation matrix.
        It includes right-handed to left-handed conversions.

        Parameters
        ----------
        rpy : list [roll, pitch, yaw]
            rpy is roll, pitch, and yaw.

        Returns
        -------
        R : numpy.ndarray
            Rotation matrix in 3 dimensions.
        """
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
        """
        This function converts the coordinate system from right-handed to left-handed.
        Used to convert from motion capture to the robot's coordinate system.
        It should be changed according to the user's environment.
        Examples of changes include the need to change the sign or rearrange the order.

        Parameters
        ----------
        orientation : class 'geometry_msgs.msg._Quaternion.Quaternion'
            Orientations of objects obtained by motion capture.

        Returns
        -------
        ori : class 'geometry_msgs.msg._Quaternion.Quaternion'
            Orientation of objects in the robot's coordinate system.
        """
        ori = copy.deepcopy(orientation)
        ori.x = -orientation.x
        ori.y = orientation.z
        ori.z = orientation.y
        ori.w = -orientation.w
        return ori

    def transform(self, local_pose, offset):
        """
        This function converts the coordinate system from right-handed to left-handed.
        Used to convert from motion capture to the robot's coordinate system.
        It should be changed according to the user's environment.
        Examples of changes include the need to change the sign or rearrange the order.

        Parameters
        ----------
        local_pose : class 'geometry_msgs.msg._Pose.Pose'
            Posture on motion capture.

        offset : list
            Offset between the motion capture origin and the Rviz origin.

        Returns
        -------
        pose : class 'geometry_msgs.msg._Pose.Pose'
            Attitude of the robot (Rviz) in the coordinate axes.
        """
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
        
