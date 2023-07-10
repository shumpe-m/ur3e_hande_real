import numpy as np
import tf
import copy


class Transformations(object):
    def __init__(self):
        super(Transformations, self).__init__()

    def rot(self, deg):
        """
        This function calculates a rotation matrix.
        Parameters
        ----------
        deg : float
            z rotation
        Returns
        -------
        R : numpy.ndarray
            Rotation matrix in 2 dimensions.
        """
        

        Rz = np.array([[np.cos(deg), np.sin(deg)],
                    [-np.sin(deg), np.cos(deg)])

        R = Rz

        return R

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
    
    def euler_to_quaternion(self, euler = [3.140876586229683, 0.0008580159308600959, -0.0009655065200909574]):
        """
        Convert Euler Angles to Quaternion.

        Parameters
        ----------
        euler : list or class 'geometry_msgs.msg._Pose.Pose'
            The Euler angles you want to convert.

        Returns
        -------
        q : numpy.ndarray
            Quaternion values.
        """
        if type(euler) == geometry_msgs.msg._Pose.Pose:
            print("The type of variables is different.")
        elif type(euler) == list:
            q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        elif type(euler) == np.ndarray:
            euler = euler.tolist()
            q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        else:
            print("The type of variables is different.")

        return list(q)

    def quaternion_to_euler(self, quaternion):
        """
        Convert Quaternion to Euler Angles.

        Parameters
        ----------
        quaternion : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            The Quaternion you want to convert.

        Returns
        -------
        e : numpy.ndarray
            Euler Angles.
        """
        if type(quaternion) == geometry_msgs.msg._Quaternion.Quaternion:
            e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        elif type(quaternion) == list:
            e = tf.transformations.euler_from_quaternion((quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
        elif type(quaternion) == np.ndarray:
            quaternion = quaternion.tolist()
            e = tf.transformations.euler_from_quaternion((quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
        else:
            print("The type of variables is different.")

        return list(e)