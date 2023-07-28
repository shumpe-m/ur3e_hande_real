#!/usr/bin/env python3
from __future__ import print_function

import sys
import time
import copy
import rospy
import datetime

import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
import seaborn as sns

import geometry_msgs.msg
from ur_control_moveit import arm, ur_gripper_controller, ft_sensor, ur_ros
from motion_capture.get_object_pose import GetChikuwaPose, GetShrimpPose, GetEggplantPose, GetGreenPapperPose, GetJigPose
from camera import get_camera_pose
from utils import transformations, random_pose, state_plot


class UrControl(object):
    def __init__(self):
        super(UrControl, self).__init__()
        # ur control
        self.arm_control = arm.ArmControl("arm")
        self.gripper_control = ur_gripper_controller.GripperController()
        self.rp = random_pose.RandomPose()
        self.ft = ft_sensor.FtMessage()
        self.ur_ros = ur_ros.URRos()
        # object pose class
        self.gcp = GetChikuwaPose()
        self.gsp = GetShrimpPose()
        self.gep = GetEggplantPose()
        self.ggp = GetGreenPapperPose()
        self.gjp = GetJigPose()
        # self.gcamp = GetCameraPose()
        self.gcamp = get_camera_pose.GetCameraPose()
        self.gdi = get_camera_pose.GetDepthImg()
        self.gci = get_camera_pose.GetColorImg()
        self.gdinfo = get_camera_pose.GetDepthInfo()
        # utils
        self.trf = transformations.Transformations()
        self.plot = state_plot.PlotPose()
        # 
        # self.forward_basic_joint = [0.9419943318766126, -1.4060059478330746, 1.4873566760577779, -1.6507993112633637, -1.5705307531751274, -0.629176246773139]
        self.forward_basic_joint = [0.9827040433883667, -1.7932526073851527, 1.555544678364889, -1.3328328293612977, -1.5719154516803187, -0.5877450148211878]
        # self.backward_basic_joint = [-2.234010390909684, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139]
        self.backward_basic_joint = [-2.144214932118551, -1.8263160190977992, 1.5622089544879358, -1.3076815468123932, -1.5734213034259241, -0.576937500630514]
        self.mid_basic_joint = [0, -1.3927192120354697, 1.472050134256044, -1.65123476873738, -1.5690119071493065, -0.629176246773139]
        self.forward_basic_anguler = [3.14, 0, -1.57]
        self.backward_basic_anguler = [3.14, 0, -1.57]

        self.gripper_control.rq_reset()
        self.gripper_control.rq_activate()
        self.gripper_control.rq_init_gripper(speed=180, force=1)

        self.obj_name = ["Chikuwa", "Shrimp", "Eggplant", "Green_papper"]
        self.num_obj = 0
        self.goal_pose = {"Chikuwa":0, "Shrimp":0, "Eggplant":0, "Green_papper":0}
        self.order_of_placement = []
        self.gripper_width = {"Chikuwa":140, "Shrimp":140, "Eggplant":170, "Green_papper":200} # 110 106 140 170

        self.obs_state = np.array([])
        self.total_target_state = np.array([])

        self.ft.reset_ftsensor()
        self.ur_ros.play()
        rospy.sleep(2.)

        self.bridge = CvBridge()

    def go_default_pose(self, area = ""):
        if area == "forward":
            base_joint = self.forward_basic_joint
        elif area == "backward":
            base_joint = self.backward_basic_joint
        else:
            base_joint = self.mid_basic_joint
        self.arm_control.go_to_joint_state(base_joint)
        lock = self.unlock_safety()

        return lock


    def pick(self, g_pose, gr_width = 0):
        # Initialization value
        self.ft.reset_ftsensor()
        can_pick = False
        can_execute = False
        collision_detection = False
        above_pose = copy.deepcopy(g_pose)
        above_pose.position.z = 0.1
        e = self.trf.quaternion_to_euler(quaternion = g_pose.orientation)
        e = self.normalize_angles(e)
        self.gripper_control.rq_gripper_move_to(gr_width)

        # Change the default pose of the robot according to the sign of y. And add offset.
        if g_pose.position.y > 0:
            area = "forward"
            base_e = self.forward_basic_anguler
            # g_pose.position.z += 0.01 # mocap offset
            g_pose.position.z -= 0.01 # realsense offset

        else:
            area = "backward"
            base_e = self.backward_basic_anguler
            g_pose.position.z -= 0.01 # realsense offset

        self.go_default_pose(area)
        if g_pose.position.z <= -0.0851:
            g_pose.position.z = -0.084
            

        # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
        q = self.trf.euler_to_quaternion(euler = [-e[0] + base_e[0], e[1] + base_e[1], -e[2] + base_e[2]])
        can_execute, _ = self.arm_control.go_to_pose(pose = above_pose, ori = q, vel_scale = 0.55)
        if can_execute:
            # Move over the target object.
            g_pose.position.z += 0.04
            can_execute, collision_detection = self.arm_control.go_to_pose(pose = g_pose, ori = q, vel_scale = 0.4)

            # Moves to the position of the target object.
            g_pose.position.z -= 0.04
            can_execute, _ = self.arm_control.go_to_pose(pose = g_pose, ori = q, vel_scale = 0.015) if can_execute else False, False

            # Close the gripper.
            self.gripper_control.rq_gripper_move_to(250)
            rospy.sleep(0.2)
            gr_state = self.gripper_control.rq_gripper_position()
            # print(gr_state)
            can_pick = True if gr_state < 240 else False

            # Lift the target object slightly.
            g_pose.position.z += 0.05
            can_execute, _ = self.arm_control.go_to_pose(pose = g_pose, ori = q, vel_scale = 0.4) if can_execute else False, False

            # Lift the target object.
            can_execute, _ = self.arm_control.go_to_pose(pose = above_pose, ori = q, vel_scale = 0.4) if can_execute else False, False

        self.go_default_pose(area)
        if can_execute == False:
            print("No plan")
            print("pose : \n", g_pose)
            print("degree : \n", list(self.trf.quaternion_to_euler(quaternion = q)))
        elif can_pick == False:
            print("picking failed")

        lock = self.unlock_safety()

        return can_execute and can_pick, lock, collision_detection

    def place(self, goal_pose, offset = 0):
        # Initialization value
        self.ft.reset_ftsensor()
        can_execute = False
        g_pose = copy.deepcopy(goal_pose)
        above_pose = copy.deepcopy(goal_pose)
        above_pose.position.z = 0.1
        e = self.trf.quaternion_to_euler(quaternion = g_pose.orientation)
        e = self.normalize_angles(e)
        
        # Change the default pose of the robot according to the sign of y. And add offset.
        if g_pose.position.y > 0:
            area = "forward"
            base_e = self.forward_basic_anguler
            # offset
            # g_pose.position.z += 0.01 + offset # mocap offset
            g_pose.position.z += 0.01 + offset # realsense offset

        else:
            area = "backward"
            base_e = self.backward_basic_anguler
            # offset
            # g_pose.position.z += offset # mocap offset
            g_pose.position.z += 0.01 + offset # realsense offset
        self.go_default_pose(area)
        if g_pose.position.z <= -0.0851:
            g_pose.position.z = -0.084 + 0.001

        # Conversion to Euler angles to match the base attitude (base_e) and object attitude.
        q = self.trf.euler_to_quaternion(euler = [-e[0] + base_e[0], e[1] + base_e[1], -e[2] + base_e[2]])
        can_execute, _ = self.arm_control.go_to_pose(pose = above_pose, ori = q, vel_scale = 0.55)
        if can_execute:
            # Move over the target object.
            g_pose.position.z += 0.04
            can_execute, collision_detection = self.arm_control.go_to_pose(pose = g_pose, ori = q, vel_scale = 0.4)

            # Moves to the position of the target object.
            g_pose.position.z -= 0.04
            can_execute, _ = self.arm_control.go_to_pose(pose = g_pose, ori = q, vel_scale = 0.018) if can_execute else False, False

            # Open the gripper.
            gr_state = self.gripper_control.rq_gripper_position()
            self.gripper_control.rq_gripper_move_to(gr_state - 70) # 40
            rospy.sleep(0.3)

            # Lift the arm.
            can_execute, _ = self.arm_control.go_to_pose(pose = above_pose, ori = q, vel_scale = 0.4) if can_execute else False, False

        self.go_default_pose(area)
        if can_execute == False:
            print("No plan")
            print("pose : \n", g_pose)
            print("degree : \n", list(self.trf.quaternion_to_euler(quaternion = q)))
            # self.gripper_control.rq_gripper_move_to(255)

        lock = self.unlock_safety()
        
        return can_execute, lock
    
    def mocap_pick_and_place(self):
        self.set_goal_pose()
        for i in range(1):
            # stack target state
            for obj in self.order_of_placement:
                target_pose = self.goal_pose[obj]
                id = self.obj_name.index(obj)
                target_state = np.array([[id, target_pose.position.x, target_pose.position.y, target_pose.position.z,
                                        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]])
                if self.total_target_state.shape == (0,):
                    self.total_target_state = target_state
                else:
                    self.total_target_state = np.block([[self.total_target_state], [target_state]])

            #pick and place
            for obj in self.order_of_placement:
                self.ft.reset_ftsensor()
                self.arm_control.go_to_joint_state(self.mid_basic_joint)
                current_pose = self.get_current_pose(obj)
                self.gripper_control.rq_gripper_move_to(0)
                pick_success = self.pick(current_pose, int(self.gripper_width[obj]))
                # print("pick_success: ", pick_success)
                if pick_success:
                    place_success = self.place(self.goal_pose[obj], 0.005)
                    # print("place_success", place_success)
                self.gripper_control.rq_gripper_move_to(0)
                self.go_default_pose(area="")
                current_pose = self.get_current_pose(obj)
                # print("<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
                # print(obj)
                # print("target pose : \n", self.goal_pose[obj])
                # print("current pose : \n", current_pose)
                # print("################################")

                # stack current state
                id = self.obj_name.index(obj)
                state = np.array([[id, current_pose.position.x, current_pose.position.y, current_pose.position.z,
                                current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]])
                if self.obs_state.shape == (0,):
                    self.obs_state = state
                else:
                    self.obs_state = np.block([[self.obs_state], [state]])
            # self.self_reset()
            input("Disassemble the object in the opposite area.\n When you are done, press 'Enter key'.")
            # self.self_reset()
            # rospy.sleep(3.)

        # save state
        now = datetime.datetime.now()
        filename_t = 'target_state_' + now.strftime('%Y%m%d_%H%M%S')
        filename_o = 'obs_state_' + now.strftime('%Y%m%d_%H%M%S')
        # path_t = Path.cwd() / "data" / filename_t
        # path_o = Path.cwd() / "data" / filename_o
        path_t = "/root/ur3e_hande_real/data/" + filename_t
        path_o = "/root/ur3e_hande_real/data/" + filename_o
        np.save(path_t, self.total_target_state)
        np.save(path_o, self.obs_state)

        # plot state
        self.plot.plot_p(self.total_target_state, self.obs_state)

    def self_reset(self):
        dish_pose = self.gjp.get_pose()
        self.gripper_control.rq_gripper_move_to(100)
        self.pick(dish_pose, 240)
        
        area = "forward" if dish_pose.position.y > 0 else "backward"
        self.go_default_pose(area)
        if area == "forward":
            reset_pose = [0.13, 0.38, 0.26]
            replace_pose = [-0.105, -0.35, 0.035]
            q = self.trf.euler_to_quaternion(euler = [3.14, 0, 1.57])
            e = [0, 0, 0]
        else:
            reset_pose = [-0.145, -0.38, 0.26]
            replace_pose = [0.095, 0.35, 0.035]
            q = self.trf.euler_to_quaternion(euler = [3.14, 0, -1.57])
            e = [0, 0, 3.14]
        self.arm_control.go_to_pose(pose = reset_pose, ori = q, vel_scale = 0.30)

        joint = self.arm_control.get_current_joint()
        joint[3] -= 0.8
        self.arm_control.go_to_joint_state(joint, vel_scale=0.3)

        self.go_default_pose("")
        q = self.trf.euler_to_quaternion(euler = e)
        dish_pose.position.x = replace_pose[0]
        dish_pose.position.y = replace_pose[1]
        dish_pose.position.z = replace_pose[2]
        dish_pose.orientation.x = q[0]
        dish_pose.orientation.y = q[1]
        dish_pose.orientation.z = q[2]
        dish_pose.orientation.w = q[3]
        self.place(dish_pose, 0.015)
        self.gripper_control.rq_gripper_move_to(0)
        
    def unlock_safety(self):
        lock = False
        safety_mode = self.ur_ros.get_safetymsg()
        if str(safety_mode)=="mode: 3":
            lock = True
            self.arm_control.clear_targets
            self.ur_ros.unlock_safety()
            self.arm_control.safe_action()

        return lock
    
    def take_images(self, img_type="depth", min=0., max=1.):
        rospy.sleep(0.05)
        camera_pose = self.gcamp.get_link_pose()
        depth_info = self.gdinfo.get_depth_info()
        if img_type=="depth":
            depth = self.gdi.get_depth()
            img = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
            img = np.array(img, dtype=np.uint16)
            img = img.clip(min * 1000, max * 1000) - min * 1000
            img = img * 255 / ((max - min) * 1000)
            img = 255 - img
            img = np.where(img == 255, 0, img)
        else:
            rgb = self.gci.get_rgb()
            img = self.bridge.imgmsg_to_cv2(rgb, 'rgb8')

        return img.astype(np.uint8), camera_pose, depth_info
    
    def pixel_to_coordinate(self, img, action, camera_pose, depth_info, min=0., max=1., action_area="forward"):
        ee_pose = camera_pose
        k = depth_info
        k_inv = np.linalg.inv(k)
        u = action[0]
        v = action[1]
        z = ((255 - img[int(v), int(u)]) * ((max - min) * 1000) / 255 + min * 1000) * 0.001
        # print(z, camera_pose.pose.position.z)
        uvw = np.array([u, v, 1]) * z
        # print(z)
        xyz = np.dot(k_inv, uvw)

        q = self.trf.euler_to_quaternion(euler = [3.14, 0, -np.pi*1.5 + action[2]])
        if action_area == "forward":
            ee_pose.pose.position.x += xyz[0]
            ee_pose.pose.position.y -= xyz[1]
            ee_pose.pose.position.z = -(z - camera_pose.pose.position.z)
        elif action_area == "backward":
            ee_pose.pose.position.x -= xyz[0]
            ee_pose.pose.position.y += xyz[1]
            ee_pose.pose.position.z = -(z - camera_pose.pose.position.z)
        else:
            print("area is wrong.")
        # print(ee_pose.pose.position.z )
        ee_pose.pose.orientation.x = q[0]
        ee_pose.pose.orientation.y = q[1]
        ee_pose.pose.orientation.z = q[2]
        ee_pose.pose.orientation.w = q[3]
        
        return ee_pose.pose

    def test(self):
        # gpc = get_camera_pose.GetPointCloud()
        # xyz, rgb = gpc.get_pose()
        # print(xyz.shape)
        # print(rgb.shape)
        # point = np.concatenate([xyz, rgb], 1)
        # print(point.shape)
        camera = self.gcamp.get_link_pose()
        gdi = get_camera_pose.GetDepthImg()
        gci = get_camera_pose.GetColorImg()
        depth = gdi.get_depth()
        rgb = gci.get_rgb()

        self.bridge = CvBridge()
        depth = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
        color = self.bridge.imgmsg_to_cv2(rgb, 'rgb8')

        print(depth.shape)



        # height, width = depth.shape
        # scale = 1
        # crop_size = [int(height * scale), int(width * scale)]
        # crop_size = color.shape
        # top = int((height / 2) - (crop_size[0] / 2))
        # bottom = top + crop_size[0]
        # left = int((width / 2) - (crop_size[1] / 2))
        # right = left + crop_size[1]
        # depth = depth[top:bottom, left:right]

        depth_image = np.array(depth, dtype=np.float32)
        # # cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)

        # sns.heatmap(depth_image, vmin=0, vmax=500, cmap='jet')

        # alpha = 0.7
        # blended = cv2.addWeighted(color, alpha, depth_image, 1 - alpha, 0)

        # # 結果を表示する。
        # plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        plt.imshow(depth_image, cmap='gray')
        # plt.imshow(color, cmap='gray')
        plt.show()


        # camera
        gdinfo = get_camera_pose.GetDepthInfo()
        k = gdinfo.get_depth_info()
        k_inv = np.linalg.inv(k)
        u = float(input("x: "))
        v = float(input("y: "))
        z = depth[int(v), int(u)] * 0.001
        uvw = np.array([u, v, 1]) * z
        print(uvw)
        xyz = np.dot(k_inv, uvw)
        print(xyz)


        q = self.trf.euler_to_quaternion(euler = [3.14, 0, -np.pi*1.5])
        camera.pose.position.x += xyz[0]
        camera.pose.position.y -= xyz[1]
        camera.pose.position.z = -0.06
        can_execute = self.arm_control.go_to_pose(pose = camera.pose, ori = q, vel_scale = 0.5)


        # print(depth_image)


        # cv2.namedWindow("depth_image")

        # cv2.imshow("depth_image", depth_image)
        # cv2.waitKey(10000)






    def test2(self):
        obj = input("obj*")
        
        self.ft.reset_ftsensor()
        self.arm_control.go_to_joint_state(self.mid_basic_joint)
        current_pose = self.get_current_pose(obj)
        pick_success = self.pick(current_pose, int(self.gripper_width[obj]))

        self.gripper_control.rq_gripper_move_to(0)
        rospy.sleep(0.5)
        self.go_default_pose(area="")
        current_pose = self.get_current_pose(obj)
        print("<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")
        print(obj)
        print("current pose : \n", current_pose)
        print("degree : \n", list(self.trf.quaternion_to_euler(quaternion = current_pose.orientation)))
        print("################################")

        # self.self_reset()
        rospy.sleep(3.)


    def test3(self):
        ee_p = self.arm_control.get_current_pose()
        print(ee_p)
        camera = self.gcamp.get_link_pose()
        print(camera)
        # camera = self.gcamp.get_pose()
        


        gdi = get_camera_pose.GetDepthImg()
        gci = get_camera_pose.GetColorImg()
        depth = gdi.get_depth()
        rgb = gci.get_rgb()

        self.bridge = CvBridge()
        depth = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
        color = self.bridge.imgmsg_to_cv2(rgb, 'rgb8')
        depth_image = np.array(depth, dtype=np.float32)

        plt.imshow(depth_image, cmap='gray')
        # plt.imshow(color, cmap='gray')
        plt.show()

        gdinfo = get_camera_pose.GetDepthInfo()
        k = gdinfo.get_depth_info()
        k_inv = np.linalg.inv(k)
        u = float(input("x: "))
        v = float(input("y: "))
        z = depth[int(v), int(u)] * 0.001
        uvw = np.array([u, v, 1]) * z
        print(uvw)
        xyz = np.dot(k_inv, uvw)
        print(xyz)


        q = self.trf.euler_to_quaternion(euler = [3.14, 0, -np.pi*1.5])
        camera.pose.position.x += xyz[0]
        camera.pose.position.y -= xyz[1]
        camera.pose.position.z = -(z - camera.pose.position.z)
        can_execute = self.arm_control.go_to_pose(pose = camera.pose, ori = q, vel_scale = 0.5)

    # def find_defoult_joint(self):
    #     ee_p = self.arm_control.get_current_pose()
    #     print(ee_p)
    #     camera = self.gcamp.get_link_pose()
    #     print(camera)
    #     # camera = self.gcamp.get_pose()
    #     q = self.trf.euler_to_quaternion(euler = [3.14, 0, -np.pi*1.5])
    #     pose = [0.00 + ee_p.position.x - camera.pose.position.x,
    #             0.35 + ee_p.position.y - camera.pose.position.y,
    #             0.20]
    #     print(pose)
    #     can_execute = self.arm_control.go_to_pose(pose = pose, ori = q, vel_scale = 0.5)


    #     gdi = get_camera_pose.GetDepthImg()
    #     gci = get_camera_pose.GetColorImg()
    #     depth = gdi.get_depth()
    #     rgb = gci.get_rgb()

    #     self.bridge = CvBridge()
    #     depth = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
    #     color = self.bridge.imgmsg_to_cv2(rgb, 'rgb8')
    #     depth_image = np.array(depth, dtype=np.float32)

    #     plt.imshow(depth_image, cmap='gray')
    #     # plt.imshow(color, cmap='gray')
    #     plt.show()

    #     joint = self.arm_control.get_current_joint()
    #     print("joint: ", joint)


    def roop_test(self):
        filename_flont_pose = "flont_pose_20230507_054844.npy"
        filename_back_pose = "back_pose_20230507_055559.npy"
        i = 0
        while True:
            print("iter : ", i)
            self.order_of_placement = []
            path = "/root/ur3e_hande_real/data/" + filename_flont_pose if i%2 == 0 else "/root/ur3e_hande_real/data/" + filename_back_pose
            goal_state = np.load(path).tolist()

            for odx in range(4):
                choose_obj = self.obj_name[int(goal_state[odx][0])]

                pose = geometry_msgs.msg.Pose()
                pose.position.x = goal_state[odx][1]
                pose.position.y = goal_state[odx][2]
                pose.position.z = goal_state[odx][3]

                pose.orientation.x = goal_state[odx][4]
                pose.orientation.y = goal_state[odx][5]
                pose.orientation.z = goal_state[odx][6]
                pose.orientation.w = goal_state[odx][7]
                self.goal_pose[choose_obj] = pose
                self.order_of_placement.append(choose_obj)
                
            self.test()
            i += 1


    def get_current_pose(self, object_name):
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


    def save_goal_pose(self, filename):
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

        for obj in self.order_of_placement:
            target_pose = self.goal_pose[obj]
            id = self.obj_name.index(obj)
            target_state = np.array([[id, target_pose.position.x, target_pose.position.y, target_pose.position.z,
                                    target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]])
            if self.total_target_state.shape == (0,):
                self.total_target_state = target_state
            else:
                self.total_target_state = np.block([[self.total_target_state], [target_state]])
        now = datetime.datetime.now()
        filename_t = filename + now.strftime('%Y%m%d_%H%M%S')
        path_t = "/root/ur3e_hande_real/data/" + filename_t
        np.save(path_t, self.total_target_state)
        print("==============================================================\n")


    def normalize_angles(self, obj_e):
        max_ang = 0.40 # 30 degree
        for i_dx in range(2):
            if abs(obj_e[i_dx]) >= 0.70:
                sign = np.sign(obj_e[i_dx])
                # obj_e[i_dx] += -1.57 * sign
                obj_e[i_dx] = 0
            if abs(obj_e[i_dx]) >= max_ang:
                sign = np.sign(obj_e[i_dx])
                obj_e[i_dx] = max_ang * sign

        return obj_e


def main():
    try:
        action = UrControl()
        action.go_default_pose(area="forward")


        # action.mocap_pick_and_place()
        # action.self_reset()
        # action.test2()


        # action.save_goal_pose("back_pose_")

        # action.mocap_pick_and_place()
        # print("backward")
        # action.self_reset()

        # action.roop_test()
        action.test3()

        # action.go_default_pose(area="")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
