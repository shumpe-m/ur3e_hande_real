#!/usr/bin/env python3
from __future__ import print_function

import sys
import time
import copy
import math
import rospy

from ur_control_moveit import arm, ur_gripper_controller, random_pose, ft_sensor


class Ur_control(object):
    def __init__(self):
        super(Ur_control, self).__init__()
        self.arm_control = arm.Arm_control("arm")
        # self.gripper_control = gripper.Gripper_control()
        self.random_pose = random_pose.Random_pose()
        self.ft_sensor = ft_sensor.FT_message()
        self.forward_basic_joint = [0.9419943318766126, -1.4060059478330746, 1.4873566760577779, -1.6507993112633637, -1.5705307531751274, -0.629176246773139] # [1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.backward_basic_joint = [-2.234010390909684, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139] # [-1.57, -1.57, 1.26, -1.57, -1.57, 0]
        self.forward_basic_anguler = [3.14, 0, 0]
        self.backward_basic_anguler = [3.14, 0, 3.14]

    def go_default_pose(self, area):
        if area == "forwrad":
            base_joint = self.forward_basic_joint
        else:
            base_joint = self.backward_basic_joint
        self.arm_control.go_to_joint_state(base_joint)

    def pick(self, goal_pose):
        if goal_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
        self.go_default_pose(area)

        pose = copy.deepcopy(goal_pose)
        # print("pick: \n", pose)
        e = list(self.arm_control.quaternion_to_euler(quaternion = pose.orientation))
        q = self.arm_control.euler_to_quaternion(euler = [base_e[0] + e[0], base_e[1] + e[1], base_e[2] + e[2]])
        pose.position.z = 0.35
        pick_success = self.arm_control.go_to_pose(pose = pose, ori = q.tolist())
        if pick_success:
            goal_pose.position.z = 0.15 # or goal_pose.position.z
            pick_success = self.arm_control.go_to_pose(pose = goal_pose, ori = q.tolist())
            rospy.sleep(0.5)
            self.arm_control.go_to_pose(pose = pose, ori = q.tolist())
        
        self.go_default_pose(area)

        if pick_success == False:
            print(pose, q.tolist())

        return pick_success

    def place(self, goal_pose):
        if goal_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
        self.go_default_pose(area)

        pose = copy.deepcopy(goal_pose)
        # print("place: \n", pose)
        e = list(self.arm_control.quaternion_to_euler(quaternion = pose.orientation))
        q = self.arm_control.euler_to_quaternion(euler = [base_e[0] + e[0], base_e[1] + e[1], base_e[2] + e[2]])
        pose.position.z = 0.35
        place_success = self.arm_control.go_to_pose(pose = pose, ori = q.tolist())
        if place_success:
            goal_pose.position.z = 0.15 # or goal_pose.position.z
            place_success = self.arm_control.go_to_pose(pose = goal_pose, ori = q.tolist())
            rospy.sleep(0.5)
            self.arm_control.go_to_pose(pose = pose, ori = q.tolist())

        self.go_default_pose(area)

        if place_success == False:
            print(pose, q.tolist())

        return place_success

    def pick_and_place(self, pick_area="forwrad", place_area="backwrad"):
        if pick_area == "forwrad":
            pick_area = "bin1"
        else:
            pick_area = "bin2"

        if place_area == "backwrad":
            place_area = "dish2"
        else:
            place_area = "dish1"

        pick_pose = self.random_pose.random_pose(area = pick_area)
        pick_pose = self.arm_control.chage_type(pick_pose)
        place_pose = self.random_pose.random_pose(area = place_area)
        place_pose = self.arm_control.chage_type(place_pose)


        pick_success = self.pick(pick_pose)
        print("pick_success: ", pick_success)
        if pick_success:
            place_success = self.place(place_pose)
            print("place_success", place_success)
            self.place(place_pose)


    def gripper_action(self):
        self.gripper_control.gripper_open()
        self.gripper_control.gripper_close()

    
    def self_reset(self, area="forwrad"):
        self.go_default_pose(area)
        sweep_posi = []
        rospy.sleep(0.2)
        if area == "forwrad":
            sweep_posi = [[0.2, 0.35, 0.35],
                         [0.2, 0.35, 0.135],
                         [-0.075, 0.35, 0.135],
                         [0.2, 0.27, 0.35],
                         [0.2, 0.27, 0.135],
                         [-0.075, 0.27, 0.135]]
            base_e = self.forward_basic_anguler
        elif area == "backwrad":
            sweep_posi = [[-0.21, -0.35, 0.35],
                         [-0.21, -0.35, 0.135],
                         [0.075, -0.35, 0.135],
                         [-0.21, -0.27, 0.35],
                         [-0.21, -0.27, 0.135],
                         [0.075, -0.27, 0.135]]
            base_e = self.backward_basic_anguler
        q = self.arm_control.euler_to_quaternion(euler = base_e)
        if sweep_posi != []:
            self.arm_control.go_to_pose(pose = sweep_posi[0], ori = q.tolist())
            self.arm_control.go_to_pose(pose = sweep_posi[1], ori = q.tolist())
            self.arm_control.go_to_pose(pose = sweep_posi[2], ori = q.tolist())
            self.go_default_pose(area)
            rospy.sleep(0.2)
            self.arm_control.go_to_pose(pose = sweep_posi[3], ori = q.tolist())
            self.arm_control.go_to_pose(pose = sweep_posi[4], ori = q.tolist())
            self.arm_control.go_to_pose(pose = sweep_posi[5], ori = q.tolist())
            self.go_default_pose(area)

        self.go_default_pose(area)

    def joint_test(self):
        joint = copy.deepcopy(self.arm_control.get_current_joint())
        print(self.arm_control.get_current_joint())
        joint[0] += math.pi / 10 # 1.4
        rospy.sleep(0.2)
        self.arm_control.go_to_joint_state(joint)
        print(self.arm_control.get_current_joint())

    def ft_test(self):
        print(self.ft_sensor.get_ft_message())
        rospy.sleep(0.3)
        # model name of picking target
        target_model = "box3"
        area = "bin1"

        if area == "dish1" or area == "bin1":
            pick_basic_joint = self.forward_basic_joint
            place_basic_joint = self.backward_basic_joint
        elif area == "dish2" or area == "bin2":
            pick_basic_joint = self.backward_basic_joint
            place_basic_joint = self.forward_basic_joint
        self.arm_control.go_to_joint_state(pick_basic_joint)

        ### Rearrange the box ###
        box_pose = [-0.1075, 0.35, 0.7801]

        ### Pick ###
        pick_p = copy.deepcopy(box_pose)
        q = self.arm_control.euler_to_quaternion(euler = [-3.14 , 0, 0])
        pick_p[2] = 0.35
        self.arm_control.go_to_pose(pose = pick_p, ori = q.tolist())
        pick_p[2] = 0.12
        pick_success = self.arm_control.go_to_pose(pose = pick_p, ori = q.tolist())
        self.arm_control.go_to_joint_state(pick_basic_joint)

    def test(self):
        gc = ur_gripper_controller.GripperController()
        print('gripper state:', gc.rq_gripper_state_eval())
        print('gripper activated?:', gc.rq_is_gripper_activate())
        
        # gripper re-activation
        gc.rq_reset()
        gc.rq_activate()

        # set speed and force
        gc.rq_init_gripper(speed=255, force=255)
        print('gripper activated?:', gc.rq_is_gripper_activate())

        time.sleep(1)

        # repeat open and close motion
        gc.rq_gripper_open()
        gc.rq_gripper_close()
        gc.rq_gripper_open()

        gc.socket_close()


def main():
    try:
        action = Ur_control()

        action.test()

        # num_obj = 1
        # action.go_default_pose(area="forwrad")


        # start = time.time()
        # for idx in range(num_obj):
        #     print(idx+1)
        #     action.pick_and_place(pick_area = "forwrad", place_area = "forwrad")

        # action.self_reset(area = "forwrad")

        # end = time.time() - start
        # print("Time to grasp and place and sweep %d number of objects (starting in area 1): %s" % (num_obj, end))

        # start = time.time()
        # action.go_default_pose(area="backwrad")
        # end = time.time() - start
        # print("Time to turn around.: ", end)


        # start = time.time()
        # for idx in range(num_obj):
        #     print(idx+1)
        #     action.pick_and_place(pick_area = "backwrad", place_area = "backwrad")

        # action.self_reset(area = "backwrad")

        # end = time.time() - start
        # print("Time to grasp and place and sweep %d number of objects (starting in area 2): %s" % (num_obj, end))

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
