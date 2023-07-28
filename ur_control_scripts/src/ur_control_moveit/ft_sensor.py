import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

class FtMessage(object):
    def __init__(self):
        super(FtMessage, self).__init__()
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
        """
        This function waits for input as the message may be empty.

        Returns
        -------
        ft_message : class 'geometry_msgs.msg._Pose.Pose'
            FT sensor value.
        """
        while self.ft_message.header.frame_id == '':
            pass

        return self.ft_message

    def collision_avoidance(self):
        """
        This function determines that contact is made when a certain force is applied.
        It may react when lifting things, but it does not change this one.
        If it reacts other than when lifting, the value needs to be changed appropriately.

        Returns
        -------
        deetect_contact : bool
            Orientation of objects in the robot's coordinate system.
        """
        ft_msg = self.get_ft_message()
        deetect_contact = True if abs(ft_msg.wrench.force.x) >= 10 or abs(ft_msg.wrench.force.y) >= 10 or ft_msg.wrench.force.z <= -4.1 else False
        # if deetect_contact:
        #     print("############# Contacted #############\n", ft_msg.wrench)
        #     print("#####################################\n")

        return deetect_contact
        
