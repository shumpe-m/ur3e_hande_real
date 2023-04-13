import numpy as np
import tf
import copy

class Transformations(object):
    def __init__(self):
        super(Transformations, self).__init__()

    # def rot(self, rpy):
    #     """
    #     This function calculates a rotation matrix.
    #     It includes right-handed to left-handed conversions.

    #     Parameters
    #     ----------
    #     rpy : list [roll, pitch, yaw]
    #         rpy is roll, pitch, and yaw.

    #     Returns
    #     -------
    #     R : numpy.ndarray
    #         Rotation matrix in 3 dimensions.
    #     """
    #     roll = rpy[0]
    #     pitch = rpy[1]
    #     yaw = rpy[2]

    #     Rx = np.array([[1, 0, 0],
    #                 [0, np.cos(roll), np.sin(roll)],
    #                 [0, -np.sin(roll), np.cos(roll)]])

    #     Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
    #                 [0, 1, 0],
    #                 [np.sin(pitch), 0, np.cos(pitch)]])

    #     Rz = np.array([[np.cos(yaw), np.sin(yaw), 0],
    #                 [-np.sin(yaw), np.cos(yaw), 0],
    #                 [0, 0, 1]])

    #     R = Rz.T.dot(Ry.T).dot(Rx.T) # .T = transform right handed axis to left handed axis

    #     return R

    def transform_leftHanded_to_rightHanded(self, local_pose, offset):
        """
        This function converts the coordinate system from right-handed to left-handed.
        Used to convert from motion capture to the robot's coordinate system.
        It should be changed according to the user's environment.

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
        pose = copy.deepcopy(local_pose)

        pose.position.x = local_pose.position.x + offset[0]
        pose.position.y = -local_pose.position.z + offset[1]
        pose.position.z = local_pose.position.y + offset[2]

        pose.orientation.x = local_pose.orientation.z
        pose.orientation.y = -local_pose.orientation.x
        pose.orientation.z = local_pose.orientation.y
        pose.orientation.w = -local_pose.orientation.w
        return pose

        
