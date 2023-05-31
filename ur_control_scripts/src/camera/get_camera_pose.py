import geometry_msgs.msg
import tf2_geometry_msgs
import rospy
import tf2_ros

from utils.transformations import Transformations

class GetCameraPose():
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tool0_to_camera = [-0.0371687, -0.0789837, -0.0158126, 0.492351, -0.512039, 0.490376, 0.504914]

    def get_link_pose(self, link_name='camera_link'):
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
