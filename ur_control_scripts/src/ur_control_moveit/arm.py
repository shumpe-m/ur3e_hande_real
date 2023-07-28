"""
Copyright (c) 2008-2013, Willow Garage, Inc.
Copyright (c) 2015-2019, PickNik, LLC.
https://github.com/ros-planning/moveit_tutorials/blob/master/LICENSE.txt
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
except:
    from math import pi, fabs, cos, sqrt
    tau = 2.0 * pi
    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


class ArmControl(object):
    def __init__(self, name = 'arm'):
        super(ArmControl, self).__init__()

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

        # Set the end-effector link for this group
        move_group.set_end_effector_link("ur_gripper_tip_link") # or tool0
        move_group.set_goal_tolerance(0.001)

        # initialize
        self.robot = robot
        self.scene = scene
        self.move_group = move_group

        # ft sensor
        self.ft_sensor = ft_sensor.FtMessage()
        self.execute_thread_resut = False


    def go_to_joint_state(self, joint_ang, vel_scale = 0.6):
        """
        Move the joint angle to the target.

        Parameters
        ----------
        joint_ang : list
            Radian values for each joint.

        vel_scale : float
            velocity scaling factor.
        """
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(vel_scale)
        self.execute_thread_result = False
        
        move_group.set_joint_value_target(joint_ang)
        plan_success, plan, planningtime, error_code = move_group.plan()

        move_group.execute(plan, True)

        move_group.stop()
        move_group.clear_pose_targets()


    def go_to_position(self, position=[0,0,0], vel_scale = 0.2):
        """
        Move the position to the target.
        The value of orientation is the value before movement.

        Parameters
        ----------
        position : list or class 'geometry_msgs.msg._Pose.Pose'
            A position value as seen from the robot. [x, y, z]
        
        vel_scale : float
            velocity scaling factor.

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        # end effector set
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(vel_scale)
        self.execute_thread_result = False
        self.col_detection = False
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
        execute_thread = threading.Thread(target=move_group.execute, args=(plan, True))
        collision_avoidance_thread = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        execute_thread.start()
        collision_avoidance_thread.start()
        execute_thread.join()
        rospy.sleep(0.05)
        self.execute_thread_result = True
        collision_avoidance_thread.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.header.frame_id!=[], self.col_detection


    def go_to_pose(self, pose, ori = [0.0, 0.0, 0.0, 0.0], vel_scale = 0.2):
        """
        Move the pose to the target.

        Parameters
        ----------
        pose : list or class 'geometry_msgs.msg._Pose.Pose'
            A position value as seen from the robot. [x, y, z]
        
        ori : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            A orientaion value as seen from the robot. [x, y, z, w]

        vel_scale : float
            velocity scaling factor.

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        # end effector set
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(vel_scale)
        self.execute_thread_result = False
        self.col_detection = False
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
        execute_thread = threading.Thread(target=move_group.execute, args=(plan, True))
        collision_avoidance_thread = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        execute_thread.start()
        collision_avoidance_thread.start()
        execute_thread.join()
        rospy.sleep(0.05)
        self.execute_thread_result = True
        collision_avoidance_thread.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.points!=[], self.col_detection

    def reset_move(self, pose, ori = [0.0, 0.0, 0.0, 0.0], vel_scale = 0.2):
        """
        Move the pose to the target.

        Parameters
        ----------
        pose : list or class 'geometry_msgs.msg._Pose.Pose'
            A position value as seen from the robot. [x, y, z]
        
        ori : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            A orientaion value as seen from the robot. [x, y, z, w]

        vel_scale : float
            velocity scaling factor.

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        # end effector set
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(vel_scale)
        self.execute_thread_result = False
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
        move_group.execute(plan, True)
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.points!=[]

    def rot_motion(self, ori = [1.0, 0.0, 0.0, 0.0], vel_scale = 0.2):
        """
        Move the orientation to the target.
        The value of position is the value before movement.

        Parameters
        ----------
        ori : list or class 'geometry_msgs.msg._Quaternion.Quaternion'
            A orientaion value as seen from the robot. [x, y, z, w]

        vel_scale : float
            velocity scaling factor.

        Returns
        -------
        plan.joint_trajectory.header.frame_id!=[] : bool
            Whether trajectory plan generation exists.
        """
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(vel_scale)
        self.execute_thread_result = False
        self.col_detection = False
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
        plan_success, plan, planningtime, error_code = move_group.plan()
        self.ft_sensor.reset_ftsensor()
        execute_thread = threading.Thread(target=move_group.execute, args=(plan, True))
        collision_avoidance_thread = threading.Thread(target=self.stop_action, args=(self.ft_sensor.collision_avoidance,))
        execute_thread.start()
        collision_avoidance_thread.start()
        execute_thread.join()
        rospy.sleep(0.05)
        self.execute_thread_result = True
        collision_avoidance_thread.join()
        move_group.stop()
        move_group.clear_pose_targets()

        return plan.joint_trajectory.points!=[], self.col_detection

    def stop_action(self, collision_func):
        """
        A function that stops and raises the operation depending on the determination of contact.

        Parameters
        ----------
        collision_func : Function
            Function to determine contact by FT sensor value.
            Return value is a bool value Contact = True

        """
        while (not self.execute_thread_result):
            if collision_func():
                self.safe_action()
                self.col_detection = True
                break
    
    def safe_action(self):
        """
        A function that stops and raises the operation depending.
        """
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        wpose = self.move_group.get_current_pose().pose
        wpose.position.z += 0.03
        self.move_group.set_pose_target(wpose)
        self.move_group.set_max_velocity_scaling_factor(0.2)

        # Founding motion plan
        plan_success, plan, planningtime, error_code = self.move_group.plan()
        self.move_group.execute(plan, True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def clear_targets(self):
        self.move_group.stop()
        self.move_group.clear_pose_targets()


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
