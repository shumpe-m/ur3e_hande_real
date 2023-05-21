#!/usr/bin/env python3
# Python 2/3 compatibility imports
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
import tf
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from visualization_msgs.msg import Marker
import json

import numpy as np
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


class RvizSetup(object):
    def __init__(self, name = 'arm'):
        super(RvizSetup, self).__init__()

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
        planning_frame = move_group.get_planning_frame()

        move_group.set_end_effector_link("ur_gripper_tip_link")
        eef_link = move_group.get_end_effector_link()

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()

        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names



    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL

    def add_box(self, name = "", pose = [0, 0, 0], size = [0.1, 0.1, 0.1], color = [0, 1.0, 0, 1], timeout=4):
        box_name = name
        scene = self.scene

        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        #box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = pose[0]
        box_pose.pose.position.y = pose[1]
        box_pose.pose.position.z = pose[2]
        box_pose.pose.orientation.x = 0
        box_pose.pose.orientation.y = 0
        box_pose.pose.orientation.z = 1.0
        box_pose.pose.orientation.w = 0

        scene.add_box(box_name, box_pose, size=(size[0], size[1], size[2]))

        return self.wait_for_state_update(box_is_known=True, timeout=timeout)


    def add_cylinder(self, name = "", pose = [0, 0, 0], height = 0.1, radius = 0.1, color = [0, 0, 1, 1], timeout=4):
        cylinder_name = name
        scene = self.scene

        cylinder_pose = geometry_msgs.msg.PoseStamped()
        cylinder_pose.header.frame_id = "world"
        #box_pose.pose.orientation.w = 1.0
        cylinder_pose.pose.position.x = pose[0]
        cylinder_pose.pose.position.y = pose[1]
        cylinder_pose.pose.position.z = pose[2]
        cylinder_pose.pose.orientation.x = 0
        cylinder_pose.pose.orientation.y = 0
        cylinder_pose.pose.orientation.z = 1.0
        cylinder_pose.pose.orientation.w = 0

        scene.add_cylinder(cylinder_name, cylinder_pose, height = height, radius = radius)


    def add_mesh(self, name = "", pose = [0.12, 0.12, 0], size_scale = [0.01, 0.01, 0.01], timeout=4):
        mesh_name = name
        scene = self.scene
        mesh_file = '/root/catkin_ws/src/ur3e_tutorial/ur_gazebo_motion_range/models/dish/mesh/dish.stl'

        mesh_pose = geometry_msgs.msg.PoseStamped()
        mesh_pose.header.frame_id = "world"
        mesh_pose.pose.position.x = pose[0]
        mesh_pose.pose.position.y = pose[1]
        mesh_pose.pose.position.z = pose[2]
        scene.add_mesh(mesh_name, mesh_pose, mesh_file, size=(size_scale[0], size_scale[1], size_scale[2]))



def main():
    try:
        print("----------------------------------------------------------")
        setup = RvizSetup("arm")
        # setup.add_box(name = "FAKE", pose = [3, 3, 0.05], size = [0.1, 0.1, 0.1])
        table_length = [0.9, 1.5]

        setup.add_box(name = "base_box_1", pose = [0, 0, 0.0525], size = [table_length[0], 0.21, 0.104])
        setup.add_box(name = "base_box_2", pose = [(table_length[0] - 0.125) / 2, 0, 0.0525], size = [0.125, 0.4, 0.104])
        setup.add_box(name = "base_box_3", pose = [-(table_length[0] - 0.125) / 2, 0, 0.0525], size = [0.125, 0.4, 0.104])
        setup.add_box(name = "table", pose = [0, 0, 0], size = [table_length[0], table_length[1], 0.01])

        # restriction
        wall_height = 1.0
        wall_length = table_length
        setup.add_box(name = "wall1", pose = [0, -(-0.03 + wall_length[1])/2, wall_height / 2], size = [wall_length[0], 0.03, wall_height])
        setup.add_box(name = "wall2", pose = [-(-0.03 + wall_length[0])/2, 0, wall_height / 2], size = [0.03, wall_length[1], wall_height])
        setup.add_box(name = "wall3", pose = [0, (-0.03 + wall_length[1])/2, wall_height / 2], size = [wall_length[0], 0.03, wall_height])
        setup.add_box(name = "wall4", pose = [(-0.03 + wall_length[0])/2 + 0.06, 0, wall_height / 2], size = [0.03, wall_length[1], wall_height])
        setup.add_box(name = "ceiling", pose = [0, 0, wall_height], size = [table_length[0], table_length[1], 0.03])

        setup.add_box(name = "mocap1", pose = [0, -(-0.16 + wall_length[1])/2, wall_height / 2], size = [0.16, 0.1, 0.2])
        setup.add_box(name = "mocap2", pose = [0, (-0.16 + wall_length[1])/2, wall_height / 2], size = [0.16, 0.1, 0.2])

        # setup.add_mesh(name = "dish1", pose = [0.12, 0.12, 0.01], size_scale = [0.01, 0.01, 0.01])

        height = 0.002 # or 0.02
        radius = 0.18 / 2
        # setup.add_cylinder(name = "dish1", pose = [0.0925, 0.35, height/2], height = height, radius = radius)
        # # setup.add_cylinder(name = "dish2", pose = [-0.1075, 0.35, height/2], height = height, radius = radius)

        # # setup.add_cylinder(name = "dish3", pose = [0.0935, -0.35, height/2], height = height, radius = radius)
        # setup.add_cylinder(name = "dish4", pose = [-0.1065, -0.35, height/2], height = height, radius = radius)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()