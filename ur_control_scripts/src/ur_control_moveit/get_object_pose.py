import rospy
import numpy as np
import message_filters
from geometry_msgs.msg import PoseStamped
from ur_control_moveit import coordinate_transformation

class Get_object_pose(object):
    def __init__(self):
        super(Get_object_pose, self).__init__()
        # ros message
        self.sub_vector_1 = message_filters.Subscriber("/mocap_pose_topic/Chikuwa_pose", PoseStamped)
        self.sub_vector_2 = message_filters.Subscriber("/mocap_pose_topic/Shrimp_pose", PoseStamped)
        self.Chikuwa_pose = PoseStamped()
        self.Shrimp_pose = PoseStamped()

        self.queue_size = 10
        fps = 100.
        self.delay = 1 / fps * 0.5

        self.mf = message_filters.ApproximateTimeSynchronizer([self.sub_vector_1, self.sub_vector_2], self.queue_size, self.delay)
        self.mf.registerCallback(self.callbackVector)

        self.ct = coordinate_transformation.Coordinate_transformation()
        self.mocap_offset = [0.02715, 0.2977, 0.00876, 1.5708, 0.0, 3.14159] # [0.0160, 0.3077, 0.0, 0.0, 0.7071, 0.7071, 0.0]


    def callbackVector(self, msg1, msg2):
        self.Chikuwa_pose = msg1
        self.Shrimp_pose = msg2


    def get_pose(self):
        # get object pose
        while self.Chikuwa_pose.header.frame_id == '' and self.Shrimp_pose.header.frame_id == '':
            pass
        self.Chikuwa_pose = self.pose_normalization(self.Chikuwa_pose)
        self.Shrimp_pose = self.pose_normalization(self.Shrimp_pose)
        return self.Chikuwa_pose, self.Shrimp_pose

    def pose_normalization(self, msg):
        pose = PoseStamped().pose
        # pose.position.x = msg.pose.position.x
        # pose.position.y = msg.pose.position.z
        # pose.position.z = msg.pose.position.y
        pose.position = msg.pose.position
        pose.orientation = msg.pose.orientation
        pose = self.ct.transform(pose, self.mocap_offset)

        return pose






