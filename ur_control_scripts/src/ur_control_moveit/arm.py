"""
Copyright (c) 2008-2013, Willow Garage, Inc.
Copyright (c) 2015-2019, PickNik, LLC.
https://github.com/ros-planning/moveit_tutorials/blob/master/LICENSE.txt

Modifications copyright (C) 2022 Shumpe MORITA.
"""
from __future__ import print_function
from six.moves import input

import sys
import time
import copy
import threading
import tf
import actionlib
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from ur_control_moveit import ft_sensor

import numpy as np
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt
    tau = 2.0 * pi
    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


class Arm_control(object):
    def __init__(self, name = 'arm'):
        super(Arm_control, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur_planner", anonymous=False)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        self.name = name
        group_name = self.name
        move_group = moveit_commander.MoveGroupCommander(group_name)
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # Set the end-effector link for this group:
        move_group.set_end_effector_link("ur_gripper_tip_link") # or tool0

        # planning_time = 10 # [s]
        # move_group.set_planning_time(planning_time)
        move_group.set_goal_tolerance(0.005)
        move_group.set_max_velocity_scaling_factor(0.05)
        ## END_SUB_TUTORIAL

        self.robot = robot
        self.scene = scene
        self.move_group = move_group

        self.ft_sensor = ft_sensor.FT_message()
        self.thread_1_resut = False


    def go_to_joint_state(self, joint_ang):
        """
        Move the joint angle to the target.

        Parameters
        ----------
        joint_ang : list
            Radian values for each joint.
        """
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.20)
        self.thread_1_result = False
        
        move_group.set_joint_value_target(joint_ang)
        plan_success, plan, planningtime, error_code = move_group.plan()

        # self.ft_sensor.reset_ftsensor()
        # thread_1 = threading.Thread(target=move_group.execute, args=(plan, True))
        # thread_2 = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        # thread_1.start()
        # thread_2.start()
        # thread_1.join()
        # self.thread_1_result = True
        # thread_2.join()

        move_group.execute(plan, True)

        move_group.stop()
        move_group.clear_pose_targets()


    def go_to_position(self, position=[0,0,0]):
        """
        Move the position to the target.
        The value of orientation is the value before movement.

        Parameters
        ----------
        position : list or class 'geometry_msgs.msg._Pose.Pose'
            A position value as seen from the robot. [x, y, z]

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        # end effector set
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.1)
        # move_group.set_end_effector_link("ur_gripper_tip_link")
        self.thread_1_result = False

        wpose = move_group.get_current_pose().pose
        current_pose = copy.deepcopy(wpose)
        # Position set
        if type(position) == geometry_msgs.msg._Pose.Pose:
            pose = position
            wpose.position = pose.position
        elif type(position) == list:
            wpose.position.x = position[0]
            wpose.position.y = position[1]
            wpose.position.z = position[2]
        move_group.set_pose_target(wpose)
        # Founding motion plan
        plan_success, plan, planningtime, error_code = move_group.plan()
        if plan.joint_trajectory.header.frame_id == "": # No motion plan found.
            print("No motion plan found. position = ", position)
        # Execute plan
        self.ft_sensor.reset_ftsensor()
        thread_1 = threading.Thread(target=move_group.execute, args=(plan, True))
        thread_2 = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        self.thread_1_result = True
        thread_2.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.header.frame_id!=[]


    def go_to_pose(self, pose, ori = [0.0, 0.0, 0.0, 0.0]):
        """
        Move the pose to the target.

        Parameters
        ----------
        pose : list or class 'geometry_msgs.msg._Pose.Pose'
            A position value as seen from the robot. [x, y, z]
        
        ori : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            A orientaion value as seen from the robot. [x, y, z, w]

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        # end effector set
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.075)
        # move_group.set_end_effector_link("ur_gripper_tip_link")
        self.thread_1_result = False

        wpose = move_group.get_current_pose().pose
        if type(pose) == geometry_msgs.msg._Pose.Pose:
            wpose.position = pose.position
        elif type(pose) == list:
            wpose.position.x = pose[0]
            wpose.position.y = pose[1]
            wpose.position.z = pose[2]
        else:
            print("The type of variables is different: pose")

        if type(ori) == geometry_msgs.msg._Quaternion.Quaternion:
            wpose.orientation = ori
        elif type(ori) == list:
            wpose.orientation.x = ori[0]
            wpose.orientation.y = ori[1]
            wpose.orientation.z = ori[2]
            wpose.orientation.w = ori[3]
        else:
            print("The type of variables is different: ori")

        move_group.set_pose_target(wpose)

        plan_success, plan, planningtime, error_code = move_group.plan()
        self.ft_sensor.reset_ftsensor()
        thread_1 = threading.Thread(target=move_group.execute, args=(plan, True))
        thread_2 = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        self.thread_1_result = True
        thread_2.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.points!=[]

    def rot_motion(self, ori = [1.0, 0.0, 0.0, 0.0]):
        """
        Move the orientation to the target.
        The value of position is the value before movement.

        Parameters
        ----------
        ori : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            A orientaion value as seen from the robot. [x, y, z, w]

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.1)
        # move_group.set_end_effector_link("ur_gripper_tip_link")
        self.thread_1_result = False

        wpose = move_group.get_current_pose().pose
        wpose.position = pose.position
        current_pose = copy.deepcopy(wpose)

        if type(ori) == geometry_msgs.msg._Quaternion.Quaternion:
            wpose.orientation = ori
        elif type(ori) == list:
            wpose.orientation.x = ori[0]
            wpose.orientation.y = ori[1]
            wpose.orientation.z = ori[2]
            wpose.orientation.w = ori[3]
        else:
            print("The type of variables is different.")

        move_group.set_pose_target(wpose)

        # print("plan")
        plan_success, plan, planningtime, error_code = move_group.plan()
        # print("exwcute")
        self.ft_sensor.reset_ftsensor()
        thread_1 = threading.Thread(target=move_group.execute, args=(plan, True))
        thread_2 = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        self.thread_1_result = True
        thread_2.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.points!=[]

    def stop_action(self, collision_func):
        """
        A function that stops and raises the operation depending on the determination of contact.

        Parameters
        ----------
        collision_func : Function
            Function to determine contact by FT sensor value.
            Return value is a bool value Contact = True

        """
        move_group = self.move_group
        while (not self.thread_1_result):
            if collision_func():
                move_group.stop()
                move_group.clear_pose_targets()
                wpose = move_group.get_current_pose().pose
                wpose.position.z += 0.05
                move_group.set_pose_target(wpose)
                # Founding motion plan
                plan_success, plan, planningtime, error_code = move_group.plan()
                move_group.execute(plan, True)
                move_group.stop()
                move_group.clear_pose_targets()
                break
            # rospy.sleep(0.1)


    def euler_to_quaternion(self, euler = [3.140876586229683, 0.0008580159308600959, -0.0009655065200909574]):
        """
        Convert Euler Angles to Quaternion.

        Parameters
        ----------
        euler : list or class 'geometry_msgs.msg._Pose.Pose'
            The Euler angles you want to convert.

        Returns
        -------
        q : numpy.ndarray
            Quaternion values.
        """
        if type(euler) == geometry_msgs.msg._Pose.Pose:
            print("The type of variables is different.")
        elif type(euler) == list:
            q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        else:
            print("The type of variables is different.")

        return list(q)

    def quaternion_to_euler(self, quaternion):
        """
        Convert Quaternion to Euler Angles.

        Parameters
        ----------
        quaternion : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            The Quaternion you want to convert.

        Returns
        -------
        e : numpy.ndarray
            Euler Angles.
        """
        if type(quaternion) == geometry_msgs.msg._Quaternion.Quaternion:
            e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        elif type(quaternion) == list:
            e = tf.transformations.euler_from_quaternion((quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
        else:
            print("The type of variables is different.")

        return list(e)

    def get_current_pose(self, end_effector_link_name = "ur_gripper_tip_link"):
        """
        Get the current robot pose.

        Parameters
        ----------
        end_effector_link_name : str
            The Quaternion you want to convert.

        Returns
        -------
        pose : class 'geometry_msgs.msg._Pose.Pose'
            Current robot pose.
        """
        move_group = self.move_group
        move_group.set_end_effector_link(end_effector_link_name)
        pose = move_group.get_current_pose().pose
        return pose

    def get_current_joint(self):
        """
        Get the current robot joint value.

        Returns
        -------
        joint : list
            Current robot joint value.
        """
        move_group = self.move_group
        joint = move_group.get_current_joint_values()
        return joint


    def chage_type(self, pose):
        # end effector set
        move_group = self.move_group
        # move_group.set_end_effector_link("ur_gripper_tip_link")
        q = self.euler_to_quaternion(pose[3:])

        wpose = move_group.get_current_pose().pose


        wpose.position.x = pose[0]
        wpose.position.y = pose[1]
        wpose.position.z = pose[2]
        wpose.orientation.x = q[0]
        wpose.orientation.y = q[1]
        wpose.orientation.z = q[2]
        wpose.orientation.w = q[3]

        return wpose

    # def to_list(self, pose):
    #     x, y, z, _, _, _, _ = pose_to_list(pose)
    #     data = np.array([[x,y,z]])
    #     return data

    # def save_array(self, data_name, data, height, r_scale):
    #     file_name = data_name + "_height_" + str(height) + "_rscale_" + str(r_scale) 
    #     np.save(file_name, data, fix_imports=True)