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
        self.forward_basic_anguler = [3.14, 0, 1.57] # 1.57
        self.backward_basic_anguler = [3.14, 0, 3.14]

        self.gc.rq_reset()
        self.gc.rq_activate()
        self.gc.rq_init_gripper(speed=180, force=1)

    def go_default_pose(self, area = ""):
        if area == "forwrad":
            base_joint = self.forward_basic_joint
        elif area == "backwrad":
            base_joint = self.backward_basic_joint
        else:
            base_joint = self.mid_basic_joint
        self.arm_control.go_to_joint_state(base_joint)

    def pick(self, goal_pose):
        self.gc.rq_gripper_move_to(0)
        if goal_pose.position.y > 0:
            area = "forwrad"
            base_e = self.forward_basic_anguler
        else:
            area = "backwrad"
            base_e = self.backward_basic_anguler
        self.go_default_pose(area)

        ### test ###
        # goal_pose.position.x = 0
        # goal_pose.position.y = 0.35
        # po=self.arm_control.get_current_pose()
        # e = list(self.arm_control.quaternion_to_euler(quaternion = po.orientation))
        # print(e)
        ### end ###

        pose = copy.deepcopy(goal_pose)
        # print("pick: \n", pose)
        e = list(self.arm_control.quaternion_to_euler(quaternion = pose.orientation))
        q = self.arm_control.euler_to_quaternion(euler = [base_e[0] + e[0], base_e[1] + e[1], base_e[2] + e[2]])
        # q = self.arm_control.euler_to_quaternion(euler = base_e)
        pose.position.z = 0.20
        pick_success = self.arm_control.go_to_pose(pose = pose, ori = q.tolist())
        if pick_success:
            goal_pose.position.z += 0.007 # or goal_pose.position.z 0.02
            pick_success = self.arm_control.go_to_pose(pose = goal_pose, ori = q.tolist())
            rospy.sleep(0.2)
            self.gc.rq_gripper_move_to(70)
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
        pose.position.z = 0.20
        place_success = self.arm_control.go_to_pose(pose = pose, ori = q.tolist())
        if place_success:
            goal_pose.position.z = 0.05 # or goal_pose.position.z
            place_success = self.arm_control.go_to_pose(pose = goal_pose, ori = q.tolist())
            rospy.sleep(0.2)
            self.gc.rq_gripper_move_to(1)
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

        pick_pose = self.rp.random_pose(area = pick_area)
        pick_pose = self.arm_control.chage_type(pick_pose)
        place_pose = self.rp.random_pose(area = place_area)
        place_pose = self.arm_control.chage_type(place_pose)


        pick_success = self.pick(pick_pose)
        print("pick_success: ", pick_success)
        if pick_success:
            place_success = self.place(place_pose)
            print("place_success", place_success)


    def gripper_action(self):
        self.gripper_control.gripper_open()
        self.gripper_control.gripper_close()

    
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
        print(self.ft.get_ft_message())
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
        # po=self.arm_control.get_current_pose()
        # print(po)
        # e = list(self.arm_control.quaternion_to_euler(quaternion = po.orientation))
        # print(e)
        
        # jo=self.arm_control.get_current_joint()
        # print(jo)


        pick_area = "bin1"
        place_area = "dish1"
        num_obj = 2

        for obj_dx in range(num_obj):
            self.arm_control.go_to_joint_state(self.mid_basic_joint)
            p1, p2 = self.gop.get_pose()
            # print("Chikuwa: ", p1)
            # print("shrimp: ", p2)
            pick_pose = p1 if obj_dx == 0 else p2
            place_pose = self.rp.random_pose(area = place_area)
            place_pose = self.arm_control.chage_type(place_pose)


            pick_success = self.pick(pick_pose)
            print("pick_success: ", pick_success)
            if pick_success:
                place_success = self.place(place_pose)
                print("place_success", place_success)

            self.gc.rq_gripper_move_to(1)


        


def main():
    try:
        action = Ur_control()

        # action.go_default_pose(area="forwrad")
        action.test()

        # num_obj = 1
        # action.go_default_pose(area="forwrad")


        # # start = time.time()
        # for idx in range(num_obj):
        #     print(idx+1)
        #     action.pick_and_place(pick_area = "forwrad", place_area = "forwrad")

        # # action.self_reset(area = "forwrad")

        # # end = time.time() - start
        # # print("Time to grasp and place and sweep %d number of objects (starting in area 1): %s" % (num_obj, end))

        # # start = time.time()
        # action.go_default_pose(area="backwrad")
        # # end = time.time() - start
        # # print("Time to turn around.: ", end)


        # # start = time.time()
        # for idx in range(num_obj):
        #     print(idx+1)
        #     action.pick_and_place(pick_area = "backwrad", place_area = "backwrad")

        # # action.self_reset(area = "backwrad")

        # # end = time.time() - start
        # # print("Time to grasp and place and sweep %d number of objects (starting in area 2): %s" % (num_obj, end))

        action.go_default_pose(area="")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
