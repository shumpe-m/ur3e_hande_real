import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

class FT_message(object):
    def __init__(self):
        super(FT_message, self).__init__()
        # ros service
        rospy.wait_for_service("/ur_hardware_interface/zero_ftsensor")
        self.zero_ftsensor = rospy.ServiceProxy("/ur_hardware_interface/zero_ftsensor", Trigger)
        # ros message
        self.sub_vector = rospy.Subscriber("/wrench", WrenchStamped, self.callbackVector)
        self.ft_message = WrenchStamped()


    def callbackVector(self, msg):
        self.ft_message = msg

    def reset_ftsensor(self):
        srv_msg = self.zero_ftsensor()

    def get_ft_message(self):
        while self.ft_message.header.frame_id == '':
            pass

        return self.ft_message

    def collision_avoidance(self):
        ft_msg = self.get_ft_message()
        collision_flag = True if abs(ft_msg.wrench.force.x) >= 40 or abs(ft_msg.wrench.force.y) >= 40 or ft_msg.wrench.force.z >= 40 else False
        # print(ft_msg.wrench.force)
        if collision_flag:
            print("############# Contacted #############\n", ft_msg.wrench)


        return collision_flag