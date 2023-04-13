#!/usr/bin/env python3
from __future__ import print_function

import sys
import time
import copy
import math
import rospy
import numpy as np

from ur_control_moveit import arm, ur_gripper_controller, random_pose, ft_sensor
from motion_capture.get_object_pose import Get_chikuwa_pose, Get_shrimp_pose, Get_eggplant_pose, Get_green_papper_pose


class Ur_control(object):
    def __init__(self):
        super(Ur_control, self).__init__()

        self.arm_control = arm.Arm_control("arm")
        self.gripper_control = ur_gripper_controller.GripperController()
        self.rp = random_pose.Random_pose()
        self.ft = ft_sensor.FT_message()
        # object pose class
        self.gcp = Get_chikuwa_pose()
        self.gsp = Get_shrimp_pose()
        self.gep = Get_eggplant_pose()
        self.ggp = Get_green_papper_pose()

        self.forward_basic_joint = [0.9419943318766126, -1.4060059478330746, 1.4873566760577779, -1.6507993112633637, -1.5705307531751274, -0.629176246773139] # [1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.backward_basic_joint = [-2.234010390909684, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139] # [-1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.mid_basic_joint = [0, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139] # [-1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.forward_basic_anguler = [3.14, 0, -1.57]
        self.backward_basic_anguler = [3.14, 0, -1.57]

        self.gripper_control.rq_reset()
        self.gripper_control.rq_activate()
        self.gripper_control.rq_init_gripper(speed=180, force=1)

        self.obj_name = ["Chikuwa", "Shrimp", "Eggplant", "Green_papper"]
        self.num_obj = 0
        self.goal_pose = {"Chikuwa":0, "Shrimp":0, "Eggplant":0, "Green_papper":0}
        self.order_of_placement = []
        self.gripper_width = {"Chikuwa":255*0.6, "Shrimp":255*0.6, "Eggplant":255*0.7, "Green_papper":255*0.85}

    def go_default_pose(self, area = ""):
        if area == "forwrad":
            base_joint = self.forward_basic_joint
        elif area == "backwrad":
            base_joint = self.backward_basic_joint
        else:
            base_joint = self.mid_basic_joint
        self.arm_control.go_to_joint_state(base_joint)


    def pick(self, g_pose, g_width = 120):
        above_pose = copy.deepcopy(g_pose)
        above_pose.position.z = 0.20
        e = self.arm_control.quaternion_to_euler(quaternion = g_pose.orientation)

        # Change the default pose of the robot according to the sign of y. And add offset.
        if g_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
            # mocap offset
            g_pose.position.z += 0.004
            # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
            q = self.arm_control.euler_to_quaternion(euler = [-e[0] + base_e[0], e[1] + base_e[1], -e[2] + base_e[2]])
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
            # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
            q = self.arm_control.euler_to_quaternion(euler = [e[0] + base_e[0], -e[1] + base_e[1], -e[2] + base_e[2]])
        self.go_default_pose(area)
        self.gripper_control.rq_gripper_move_to(0)

        # print("target_degree: \n", [e[0]*180/3.1415, e[1]*180/3.1415, e[2]*180/3.1415])


        can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q)
        if can_execute:
            # Approaching target object
            g_pose.position.z += 0.02
            can_execute = self.arm_control.go_to_pose(pose = g_pose, ori = q)
            # Move to the height of the target object.
            g_pose.position.z -= 0.019
            can_execute = self.arm_control.go_to_pose(pose = g_pose, ori = q) if can_execute else False
            rospy.sleep(0.1)
            # Close the gripper.
            self.gripper_control.rq_gripper_move_to(g_width)
            rospy.sleep(0.3)
            # Lift the target object slightly.
            g_pose.position.z += 0.03
            can_execute = self.arm_control.go_to_pose(pose = g_pose, ori = q) if can_execute else False
            # Lift the target object.
            can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q) if can_execute else False
        
        self.go_default_pose(area)
        

        if can_execute == False:
            print("No plan")
            print("pose : \n", g_pose)
            print("degree : \n", list(self.arm_control.quaternion_to_euler(quaternion = q)))


        return can_execute

    def place(self, g_pose):
        above_pose = copy.deepcopy(g_pose)
        above_pose.position.z = 0.20
        e = self.arm_control.quaternion_to_euler(quaternion = g_pose.orientation)

        # Change the default pose of the robot according to the sign of y. And add offset.
        if g_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
            # mocap offset
            g_pose.position.z += 0.004
            # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
            q = self.arm_control.euler_to_quaternion(euler = [-e[0] + base_e[0], e[1] + base_e[1], -e[2] + base_e[2]])
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
            # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
            q = self.arm_control.euler_to_quaternion(euler = [e[0] + base_e[0], -e[1] + base_e[1], -e[2] + base_e[2]])

        self.go_default_pose(area)

        can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q)
        if can_execute:
            # Approaching target object position.
            g_pose.position.z += 0.04
            can_execute = self.arm_control.go_to_pose(pose = g_pose, ori = q)
            # Move to the height of the target object position.
            g_pose.position.z -= 0.026
            can_execute = self.arm_control.go_to_pose(pose = g_pose, ori = q) if can_execute else False
            rospy.sleep(0.2)
            # Open the gripper.
            self.gripper_control.rq_gripper_move_to(0)
            rospy.sleep(0.5)
            # Lift the arm.
            can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q) if can_execute else False

        self.go_default_pose(area)

        if can_execute == False:
            print("No plan")
            print("pose : \n", g_pose)
            print("degree : \n", list(self.arm_control.quaternion_to_euler(quaternion = q)))

        return can_execute

    def pick_and_place(self):
        self.set_goal_pose()

        for obj_dx in range(self.num_obj):
            self.arm_control.go_to_joint_state(self.mid_basic_joint)
            current_pose = self.get_current_pose()

            pick_success = self.pick(current_pose[obj_dx])
            print("pick_success: ", pick_success)
            if pick_success:
                place_success = self.place(self.goal_pose[obj_dx])
                print("place_success", place_success)

            self.gripper_control.rq_gripper_move_to(0)
    
    def mocap_pick_and_place(self):
        self.set_goal_pose()

        for obj in self.order_of_placement:
            self.ft.reset_ftsensor()
            self.arm_control.go_to_joint_state(self.mid_basic_joint)
            current_pose = self.get_current_pose(obj)

            pick_success = self.pick(current_pose, int(self.gripper_width[obj]))
            print("pick_success: ", pick_success)
            if pick_success:
                place_success = self.place(self.goal_pose[obj])
                print("place_success", place_success)

            self.gripper_control.rq_gripper_move_to(0)
            current_pose = self.get_current_pose(obj)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
            print(obj)
            print("target pose : \n", self.goal_pose[obj])
            print("current pose : \n", current_pose)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<")
        self.go_default_pose(area="")

        rospy.sleep(0.5)
        # print("goal_pose: ", self.goal_pose)
        # current_pose = self.get_current_pose("all")
        # print("current_pose: ", current_pose)
        # chikuwa = self.cal_squared_error(self.goal_pose[self.obj_name[0]], current_pose[0])
        # shrimp = self.cal_squared_error(self.goal_pose[self.obj_name[1]], current_pose[1])
        # eggplant = self.cal_squared_error(self.goal_pose[self.obj_name[2]], current_pose[2])
        # green_papper = self.cal_squared_error(self.goal_pose[self.obj_name[3]], current_pose[3])

        # print("chikuwa: \n",chikuwa)
        # print("shrimp: \n",shrimp)
        # print("eggplant: \n",eggplant)
        # print("green papper: \n",green_papper)

    
    def self_reset(self, area="forwrad"):
        self.go_default_pose(area)
        sweep_posi = []
        rospy.sleep(0.2)
        if area == "forwrad":
            sweep_posi = [[0.2, 0.35, 0.23],
                         [0.2, 0.35, 0.135],
                         [-0.075, 0.35, 0.135],
                         [0.2, 0.27, 0.23],
                         [0.2, 0.27, 0.135],
                         [-0.075, 0.27, 0.135]]
            base_e = self.forward_basic_anguler
        elif area == "backwrad":
            sweep_posi = [[-0.21, -0.35, 0.23],
                         [-0.21, -0.35, 0.135],
                         [0.075, -0.35, 0.135],
                         [-0.21, -0.27, 0.23],
                         [-0.21, -0.27, 0.135],
                         [0.075, -0.27, 0.135]]
            base_e = self.backward_basic_anguler
        q = self.arm_control.euler_to_quaternion(euler = base_e)
        if sweep_posi != []:
            self.arm_control.go_to_pose(pose = sweep_posi[0], ori = q)
            self.arm_control.go_to_pose(pose = sweep_posi[1], ori = q)
            self.arm_control.go_to_pose(pose = sweep_posi[2], ori = q)
            self.go_default_pose(area)
            rospy.sleep(0.2)
            self.arm_control.go_to_pose(pose = sweep_posi[3], ori = q)
            self.arm_control.go_to_pose(pose = sweep_posi[4], ori = q)
            self.arm_control.go_to_pose(pose = sweep_posi[5], ori = q)
            self.go_default_pose(area)

        self.go_default_pose(area)

    def joint_test(self):
        joint = copy.deepcopy(self.arm_control.get_current_joint())
        print(self.arm_control.get_current_joint())
        joint[0] += math.pi / 10 # 1.4
        rospy.sleep(0.2)
        self.arm_control.go_to_joint_state(joint)
        print(self.arm_control.get_current_joint())

    def get_current_pose(self, object_name):
        ["Chikuwa", "Shrimp", "Eggplant", "Green_papper"]
        if object_name == "Chikuwa":
            return self.gcp.get_pose()
        elif  object_name == "Shrimp":
            return self.gsp.get_pose()
        elif  object_name == "Eggplant":
            return self.gep.get_pose()
        elif  object_name == "Green_papper":
            return self.ggp.get_pose()
        elif object_name == "all":
            return [self.gcp.get_pose(), self.gsp.get_pose(), self.gep.get_pose(), self.ggp.get_pose()]
        else:
            print("Invalid object name.")


    def set_goal_pose(self):
        obj = copy.copy(self.obj_name)
        print("==============================================================\n")
        print("Now you will set the target postures of the objects ONE BY ONE.\n")

        while True:
            if len(obj) == 0:
                break
            print("#----------------------------------------------------------------------------#\n")
            print("List of set objects: ", obj)
            print("\n")
            print("Place the choosing object in the workspace in the target position and orientation.")
            choose_obj = input("Choose from the list the name of the object you are about to set up, type it in and press Enter.")
            if choose_obj in obj:
                print("wait")
                rospy.sleep(2.)
                self.goal_pose[choose_obj] = self.get_current_pose(choose_obj)
                obj = [name for name in obj if name != choose_obj]
                self.order_of_placement.append(choose_obj)
                self.num_obj += 1
            elif choose_obj == "end":
                break
            else:
                print("##### Invalid object name. Choose from the list the name of the object. #####")

        print("#----------------------------------------------------------------------------#\n")
        print("Goal pose received.")
        input("Disassemble the object in the opposite area.\n When you are done, press 'Enter key'.")
        print("wait")
        rospy.sleep(3.)
        print("==============================================================\n")
        

    def test(self):
        # po=self.arm_control.get_current_pose()
        # print(po)
        # e = list(self.arm_control.quaternion_to_euler(quaternion = po.orientation))
        # print(e)
        # jo=self.arm_control.get_current_joint()
        # print(jo)
        self.set_goal_pose()

        # # pick_area = "bin1"
        # # place_area = "dish1"

        # for obj_dx in range(self.num_obj):
        #     self.arm_control.go_to_joint_state(self.mid_basic_joint)
        #     current_pose = self.get_current_pose()

        #     # print("Chikuwa: ", p1)
        #     # print("shrimp: ", p2)
        #     # place_pose = self.rp.random_pose(area = place_area)
        #     # place_pose = self.arm_control.chage_type(place_pose)


        #     pick_success = self.pick(current_pose[obj_dx])
        #     print("pick_success: ", pick_success)
        #     if pick_success:
        #         place_success = self.place(self.goal_pose[obj_dx])
        #         print("place_success", place_success)

        #     self.gripper_control.rq_gripper_move_to(0)
        # self.go_default_pose(area="")


        # rospy.sleep(0.5)
        # print("goal_pose: ", self.goal_pose)
        # current_pose = self.get_current_pose()
        # print("current_pose: ", current_pose)
        # chikuwa = self.cal_squared_error(self.goal_pose[0], current_pose[0])
        # shrimp = self.cal_squared_error(self.goal_pose[1], current_pose[1])

        # print("chikuwa: \n",chikuwa)
        # print("shrimp: \n",shrimp)

    def normalize_angles(self, obj_e, base_e):
        new_e = copy.copy(obj_e)
        sign = np.sign(obj_e[2])
        # if abs(obj_e[0]) > abs(obj_e[1]):
        #     new_e[0] = obj_e[1]
        #     new_e[1] = obj_e[0]
        
        return [-new_e[0] + base_e[0], new_e[1] + base_e[1], -new_e[2] + base_e[2]]

    def cal_squared_error(self, g_pose, current_pose):
        se_px = abs(g_pose.position.x)  - abs(current_pose.position.x)  
        se_py = abs(g_pose.position.y) - abs(current_pose.position.y)
        se_pz = abs(g_pose.position.z) - abs(current_pose.position.z) 

        se_qx = abs(g_pose.orientation.x)  - abs(current_pose.orientation.x)  
        se_qy = abs(g_pose.orientation.y)  - abs(current_pose.orientation.y) 
        se_qz = abs(g_pose.orientation.z)  - abs(current_pose.orientation.z) 
        se_qw = abs(g_pose.orientation.w)  - abs(current_pose.orientation.w) 

        return [se_px, se_py, se_pz, se_qx, se_qy, se_qz, se_qw]


def main():
    try:
        action = Ur_control()
        action.go_default_pose(area="")

        action.mocap_pick_and_place()
        # action.test()

        action.go_default_pose(area="")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
