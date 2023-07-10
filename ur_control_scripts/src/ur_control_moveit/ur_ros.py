import rospy
from ur_dashboard_msgs.msg import SafetyMode
from std_srvs.srv import Trigger

class URRos(object):
    def __init__(self):
        super(FtMessage, self).__init__()
        # ros service
        rospy.wait_for_service("/ur_hardware_interface/dashboard/unlock_protective_stop")
        self.unlock_protective_stop = rospy.ServiceProxy("/ur_hardware_interface/dashboard/unlock_protective_stop", Trigger)

        rospy.wait_for_service("/ur_hardware_interface/dashboard/play")
        self.play = rospy.ServiceProxy("/ur_hardware_interface/dashboard/play", Trigger)
        # ros message
        self.sub_vector = rospy.Subscriber("/ur_hardware_interface/safety_mode", SafetyMode, self.callbackVector)
        self.safetymsg = SafetyMode()

    def callbackVector(self, msg):
        self.safetymsg = msg

    def get_safetymsg(self):
        """
        This function waits for input as the message may be empty.

        Returns
        -------
        safetymsg : class 'ur_dashboard_msgs.msg'
            safety mode msg.
        """
        while self.safetymsg.header.frame_id == '':
            pass

        return self.safetymsg

        
