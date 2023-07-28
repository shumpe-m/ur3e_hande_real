#!/usr/bin/env python3
import os
import sys
from subprocess import Popen
import time
import argparse
import json
import pickle
import yaml
from pathlib import Path

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from manipulate import UrControl
from learning_scripts.learning.train import Train
from learning_scripts.inference.inference import Inference
from learning_scripts.inference.inference_utils import InferenceUtils
from learning_scripts.utils.param import Mode, SelectionMethod

class SelfLearning(UrControl):
   def __init__(self, previous_experience, dataset_path, percentage_secondary):
      super().__init__()
      # learning_scrpipts path
      self.ls_path = "/root/learning_data"
      with open(self.ls_path + '/config/config.yml', 'r') as yml:
         config = yaml.safe_load(yml)
      self.total_episodes = config["manipulation"]["episode"]
      self.random = config["manipulation"]["random"]
      self.min = config["depth"]["distance_min"]
      self.max = config["depth"]["distance_max"]
      
      self.inference = Inference(upper_random_pose=[config["inference"]["img_width"], config["inference"]["img_height"], 3.1415/2], ls_path=self.ls_path)
      self.load_train_model = False
      self.image_states = ["grasp", "place_b", "goal", "place_a"]
      self.percentage_secondary = percentage_secondary
      self.primary_selection_method = SelectionMethod.PowerProb
      self.secondary_selection_method = "oracle"
      self.previous_model_timestanp=""
      self.dataset_path = dataset_path

      self.gripper = [0, 25, 75]
      
      if previous_experience:
         with open(self.dataset_path, mode="rt", encoding="utf-8") as f:
            self.dataset = json.load(f)
         keys = list(self.dataset.keys())
         key = keys[-1]
         self.episode = int(key)
      else:
         self.episode = 0
         
      if self.episode < 5:
         # initialize
         self.dataset = {}
         ini_t = {}
         json_file = open(self.ls_path + '/data/datasets/learning_time.json', mode="w")
         json.dump(ini_t, json_file, ensure_ascii=False)
         json_file.close()

      self.train = Train(
         dataset_path=self.dataset_path,
         ls_path=self.ls_path,
         image_format="png",
      )

   def manipulate(self):
      data = {}
      time_data = {}
      img_type = "depth"

      only_grasp = True
      if only_grasp:
         grasp_area = "forward"
         place_area = "forward"
      else:
         grasp_area = "forward"
         place_area = "backward"

      while self.episode < self.total_episodes:
         lock_b = False
         collision_detection = False
         start = time.time()
         print("current episode: ", self.episode)
         if self.episode < self.random:
            method = SelectionMethod.Random
            method = "oracle"
         else:
            method = self.primary_selection_method if np.random.rand() > self.percentage_secondary else self.secondary_selection_method
         # method = "oracle"

         # take a before placed image
         if only_grasp:
            lock = self.go_default_pose(place_area)
            lock_b = lock_b or lock
            _, camera_pose_pb, depth_info_pb = self.take_images(img_type, self.min, self.max)
            place_b_img = cv2.imread(self.ls_path + '/data/img/' + img_type + '_place_b' + str(0) + '.png', cv2.IMREAD_UNCHANGED)
         else:
            lock = self.go_default_pose(place_area)
            lock_b = lock_b or lock
            place_b_img, camera_pose_pb, depth_info_pb = self.take_images(img_type, self.min, self.max)
         cv2.imwrite(self.ls_path + '/data/img/' + img_type + '_place_b' + str(self.episode) + '.png', place_b_img)

         # take a before grasped image
         lock = self.go_default_pose(grasp_area)
         lock_b = lock_b or lock
         grasp_img, camera_pose_g, depth_info_g = self.take_images(img_type, self.min, self.max)
         cv2.imwrite(self.ls_path + '/data/img/' + img_type + '_grasp' + str(self.episode) + '.png', grasp_img)

         # goal images
         num = np.random.randint(0,5)
         step = 0
         goal_img = cv2.imread(self.ls_path + '/data/goal/depth_goal' + str(num) + str(step) + '.png', cv2.IMREAD_UNCHANGED)

         # sampling actions
         actions = self.inference.infer(grasp_img, goal_img, method, place_images=place_b_img)

         # plt.imshow(grasp_img, cmap='gray')
         # plt.show()


         # grasp action
         pose = self.pixel_to_coordinate(grasp_img, actions["grasp"]["pose"], camera_pose_g, depth_info_g, self.min, self.max, grasp_area)
         gripper_width = self.gripper[actions["grasp"]["index"]]
         can_pick, lock, collision_detection = self.pick(pose, gr_width=gripper_width)
         lock_b = lock_b or lock

         reward = 0
         if can_pick and not lock_b:
            reward = 1
         actions["grasp"]["reward"] = reward

         if can_pick and not lock_b:
            # place action
            pose = self.pixel_to_coordinate(place_b_img, actions["place"]["pose"], camera_pose_pb, depth_info_pb, self.min, self.max, place_area)
            can_execute, lock = self.place(pose)
            lock_b = lock_b or lock
            # take a after placed image
            lock = self.go_default_pose(place_area)
            lock_b = lock_b or lock
            place_a_img, _, _ = self.take_images(img_type, self.min, self.max)
            reward = 1 
            print("### place success ###")
         else:
            # take a after placed image
            place_a_img = place_b_img
            reward = 0
         cv2.imwrite(self.ls_path + '/data/img/' + img_type + '_place_a' + str(self.episode) + '.png', place_a_img)
         actions["place"]["reward"] = reward
         self.gripper_control.rq_gripper_move_to(255)

         
         lock = self.unlock_safety()
         lock_b = lock_b or lock or collision_detection

         if lock_b == False:
            # save data
            self.dataset[str(self.episode)] = actions
            json_file = open(self.dataset_path, mode="w")
            json.dump(self.dataset, json_file, ensure_ascii=False)
            json_file.close()

            # learning
            if self.episode > self.random:
               self.load_train_model = True
            i_time = time.time() - start
            # init
            if self.episode % 20 == 0:
               self.train = Train(
                  dataset_path=self.dataset_path,
                  ls_path=self.ls_path,
                  image_format="png",
               )

            start = time.time()
            if self.episode > self.random - 1:
               self.train.run(self.load_train_model)

            l_time = time.time() - start

            t_data = [i_time, l_time]
            time_data[str(self.episode)] = t_data
            json_file = open(self.ls_path + '/data/datasets/main_time.json', mode="w")
            json.dump(time_data, json_file, ensure_ascii=False)
            json_file.close()

            print("inference_time {:.2g}s".format(i_time))
            print("learning_time {:.2g}s".format(l_time))
            print("\n")

            if reward == 1:
               grasp_area_before = grasp_area
               grasp_area = place_area
               place_area = grasp_area_before

            self.episode += 1
         else:
            self.ur_ros.unlock_safety()
            print("This episode is locked.")

parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
parser.add_argument('-e', '--previous_experience', help='Using previous experience', action='store_true')
parser.add_argument('-p', '--dataset_path', action='store', type=str, default='/root/learning_data/data/datasets/datasets.json')
parser.add_argument('-m', '--percentage_secondary_method', action='store', type=float, default=0.)
args = parser.parse_args()

learn = SelfLearning(
   previous_experience = args.previous_experience,
   dataset_path = args.dataset_path,
   percentage_secondary = args.percentage_secondary_method,
)
learn.manipulate()