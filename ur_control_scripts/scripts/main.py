#!/usr/bin/env python3
from __future__ import print_function

import sys
import time
import copy
import math
import rospy

from ur_control_moveit import arm, ur_gripper_controller, random_pose, ft_sensor, get_object_pose


class Ur_control(object):
    def __init__(self):
        super(Ur_control, self).__init__()

        self.arm_control = arm.Arm_control("arm")
        self.gc = ur_gripper_controller.GripperController()
        self.rp = random_pose.Random_pose()
        self.ft = ft_sensor.FT_message()
        self.gop = get_object_pose.Get_object_pose()

        self.forward_basic_joint = [0.9419943318766126, -1.4060059478330746, 1.4873566760577779, -1.6507993112633637, -1.5705307531751274, -0.629176246773139] # [1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.backward_basic_joint = [-2.234010390909684, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139] # [-1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.mid_basic_joint = [0, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139] # [-1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.forward_basic_anguler = [3.14, 0, 1.57]
        self.backward_basic_anguler = [3.14, 0, -1.57] # [3.14, 0, -1.57]

        self.gc.rq_reset()
        self.gc.rq_activate()
        self.gc.rq_init_gripper(speed=180, force=1)

        self.num_obj = 2
        self.goal_pose = []

    def go_default_pose(self, area = ""):
        if area == "forwrad":
            base_joint = self.forward_basic_joint
        elif area == "backwrad":
            base_joint = self.backward_basic_joint
        else:
            base_joint = self.mid_basic_joint
        self.arm_control.go_to_joint_state(base_joint)


    def pick(self, goal_pose):
        above_pose = copy.deepcopy(goal_pose)
        above_pose.position.z = 0.20
        # Change the default pose of the robot according to the sign of y.
        self.gc.rq_gripper_move_to(0)
        if goal_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
        self.go_default_pose(area)
        self.gc.rq_gripper_move_to(0)

        # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
        e = self.arm_control.quaternion_to_euler(quaternion = goal_pose.orientation)
        q = self.arm_control.euler_to_quaternion(euler = [base_e[0] + e[0], base_e[1] + e[1], base_e[2] - e[2]])

        can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q)
        if can_execute:
            # Approaching target object
            goal_pose.position.z += 0.02
            can_execute = self.arm_control.go_to_pose(pose = goal_pose, ori = q)
            # Move to the height of the target object.
            goal_pose.position.z -= 0.017
            can_execute = self.arm_control.go_to_pose(pose = goal_pose, ori = q) if can_execute else False
            rospy.sleep(0.1)
            # Close the gripper.
            self.gc.rq_gripper_move_to(120)
            rospy.sleep(0.3)
            # Lift the target object slightly.
            goal_pose.position.z += 0.03
            can_execute = self.arm_control.go_to_pose(pose = goal_pose, ori = q) if can_execute else False
            # Lift the target object.
            can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q) if can_execute else False
        
        self.go_default_pose(area)

        if can_execute == False:
            print("No plan")
            print(goal_pose, q)

        return can_execute

    def place(self, goal_pose):
        above_pose = copy.deepcopy(goal_pose)
        above_pose.position.z = 0.20
        # Change the default pose of the robot according to the sign of y.
        if goal_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
        self.go_default_pose(area)

        # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
        e = self.arm_control.quaternion_to_euler(quaternion = goal_pose.orientation)
        q = self.arm_control.euler_to_quaternion(euler = [base_e[0] + e[0], base_e[1] + e[1], base_e[2] - e[2]])
        
        can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q)
        if can_execute:
            # Approaching target object position.
            goal_pose.position.z += 0.04
            can_execute = self.arm_control.go_to_pose(pose = goal_pose, ori = q)
            # Move to the height of the target object position.
            goal_pose.position.z -= 0.025
            can_execute = self.arm_control.go_to_pose(pose = goal_pose, ori = q) if can_execute else False
            rospy.sleep(0.2)
            # Open the gripper.
            self.gc.rq_gripper_move_to(0)
            rospy.sleep(0.5)
            # Lift the arm.
            can_execute = self.arm_control.go_to_pose(pose = above_pose, ori = q) if can_execute else False

        self.go_default_pose(area)

        if can_execute == False:
            print("No plan")
            print(goal_pose.position, list(self.arm_control.quaternion_to_euler(quaternion = q)))

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

            self.gc.rq_gripper_move_to(0)

    
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

    def get_current_pose(self):
        current_pose = self.gop.get_pose()
        return current_pose

    def set_goal_pose(self):
        print("==============================================================")
        print("Now we are going to set a goal for the object.")
        input("When the goal pose of the object is complete, press 'Enter key'.")
        rospy.sleep(0.4)
        self.goal_pose = self.get_current_pose()
        print("Goal pose received.")
        input("Disassemble the object in the opposite area.\n When you are done, press 'Enter key'.")
        print("==============================================================")
        rospy.sleep(0.4)

    def test(self):
        # po=self.arm_control.get_current_pose()
        # print(po)
        # e = list(self.arm_control.quaternion_to_euler(quaternion = po.orientation))
        # print(e)
        # jo=self.arm_control.get_current_joint()
        # print(jo)
        self.set_goal_pose()

        # pick_area = "bin1"
        # place_area = "dish1"

        for obj_dx in range(self.num_obj):
            self.arm_control.go_to_joint_state(self.mid_basic_joint)
            current_pose = self.get_current_pose()

            # print("Chikuwa: ", p1)
            # print("shrimp: ", p2)
            # place_pose = self.rp.random_pose(area = place_area)
            # place_pose = self.arm_control.chage_type(place_pose)


            pick_success = self.pick(current_pose[obj_dx])
            print("pick_success: ", pick_success)
            if pick_success:
                place_success = self.place(self.goal_pose[obj_dx])
                print("place_success", place_success)

            self.gc.rq_gripper_move_to(0)

        # po.position.x = 0.00950398034465093
        # po.position.y = 0.3485885118025321
        # po.position.z = 0.07766230307384975

        # place_success = self.arm_control.go_to_pose(pose = po, ori = po.orientation)



        


def main():
    try:
        action = Ur_control()

        action.test()
        action.go_default_pose(area="")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
