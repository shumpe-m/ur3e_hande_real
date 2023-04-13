import rospy
import numpy as np
import message_filters
from geometry_msgs.msg import PoseStamped
from motion_capture import transformations

class Get_object_pose(object):
    def __init__(self):
        self.ct = transformations.Transformations()
        self.mocap_offset = [0.02879, 0.3333, -0.004, 0.0, 0.0, 0.0] #xzy  [0.02879, 0.3333, -0.005, 0.0, 0.0, 0.0]

    def wait_get_pose(self, pose_msg):
        """
        This function obtains the object's orientation from the motion capture.

        Returns
        -------
        Chikuwa_pose : class 'geometry_msgs.msg._Pose.Pose'
            Chikuwa's posture.

        Shrimp_pose : class 'geometry_msgs.msg._Pose.Pose'
            Shrimp's posture.
        """
        while pose_msg.header.frame_id == '':
            pass
        pose_msg = self.pose_normalization(pose_msg)
        return pose_msg

    def pose_normalization(self, pose_msg):
        """
        This function normalizes the coordinate axes and positions of the motion capture and robot.

        Returns
        -------
        pose : class 'geometry_msgs.msg._Pose.Pose'
            The posture of the object in the coordinate space of the robot (Rviz).
        """
        pose = PoseStamped().pose
        # pose.position.x = pose_msg.pose.position.x
        # pose.position.y = pose_msg.pose.position.z
        # pose.position.z = pose_msg.pose.position.y
        pose.position = pose_msg.pose.position
        pose.orientation = pose_msg.pose.orientation
        pose = self.ct.transform_leftHanded_to_rightHanded(pose, self.mocap_offset)

        return pose


class Get_chikuwa_pose(Get_object_pose):
    def __init__(self):
        super().__init__()
        # ros message
        self.sub_vector = rospy.Subscriber("/mocap_pose_topic/Chikuwa_pose", PoseStamped, self.callbackVector)
        self.chikuwa_pose = PoseStamped()

    def callbackVector(self, msg):
        self.chikuwa_pose = msg

    def get_pose(self):
        pose = self.wait_get_pose(self.chikuwa_pose)
        # offset mocap
        pose.position.z -= 0.003
        return pose


class Get_shrimp_pose(Get_object_pose):
    def __init__(self):
        super().__init__()
        # ros message
        self.sub_vector = rospy.Subscriber("/mocap_pose_topic/Shrimp_pose", PoseStamped, self.callbackVector)
        self.shrimp_pose = PoseStamped()

    def callbackVector(self, msg):
        self.shrimp_pose = msg

    def get_pose(self):
        pose = self.wait_get_pose(self.shrimp_pose)
        # offset mocap
        pose.position.z -= 0.004
        return pose


class Get_eggplant_pose(Get_object_pose):
    def __init__(self):
        super().__init__()
        # ros message
        self.sub_vector = rospy.Subscriber("/mocap_pose_topic/Eggplant_pose", PoseStamped, self.callbackVector)
        self.eggplamt_pose = PoseStamped()

    def callbackVector(self, msg):
        self.eggplamt_pose = msg

    def get_pose(self):
        pose = self.wait_get_pose(self.eggplamt_pose)
        return pose


class Get_green_papper_pose(Get_object_pose):
    def __init__(self):
        super().__init__()
        # ros message
        self.sub_vector = rospy.Subscriber("/mocap_pose_topic/Green_papper_pose", PoseStamped, self.callbackVector)
        self.green_papper_pose = PoseStamped()

    def callbackVector(self, msg):
        self.green_papper_pose = msg

    def get_pose(self):
        pose = self.wait_get_pose(self.green_papper_pose)
        return pose