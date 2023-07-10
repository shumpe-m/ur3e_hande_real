#!/usr/bin/env python3
import os
import sys
from subprocess import Popen
import time
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

from learning_scripts.learning.train import Train
from learning_scripts.inference.inference import Inference
from learning_scripts.inference.inference_utils import InferenceUtils
from manipulate import UrControl
from learning_scripts.utils.param import Mode, SelectionMethod
from learning_scripts.action.grasp_decision import GraspDecision
from learning_scripts.learning.train import Train


class SelfLearning(UrControl):
   def __init__(self, previous_experience, dataset_path, percentage_secondary):
      super().__init__()
      self.inference = Inference()
      self.grasp_decision = GraspDecision()
      self.load_train_model =  False
      self.image_states = ["grasp", "place_b", "goal", "place_a"]
      self.percentage_secondary = percentage_secondary
      self.primary_selection_method = SelectionMethod.Max
      self.secondary_selection_method = SelectionMethod.Prob
      self.previous_model_timestanp=""
      self.dataset_path = dataset_path
      
      if previous_experience:
         with open(self.dataset_path, mode="rt", encoding="utf-8") as f:
            self.dataset = json.load(f)
         keys = list(self.dataset.keys())
         key = keys[-2]
         self.episode = int(key)
      else:
         self.episode = 0
         
      if self.episode < 5:
         # initialize
         self.dataset = {}
         ini_t = {}
         json_file = open('./data/datasets/learning_time.json', mode="w")
         json.dump(ini_t, json_file, ensure_ascii=False)
         json_file.close()

      self.train = Train(
         image_format="png",
         dataset_path=self.dataset_path
      )

   def manipulate(self):
      data = {}
      time_data = {}

      while self.episode < 12000:
         start = time.time()
         print(self.episode)
         if self.episode < 200:
            method = SelectionMethod.Random
         else:
            method = self.primary_selection_method if np.random.rand() > self.percentage_secondary else self.secondary_selection_method

            
         # TODO: get camera images

         # grasp images
         grasp_imgs = UrControl.gcamp.get_link_pose()
         cv2.imwrite("./learning_scripts/data/img/depth_"+ self.image_states[0] + str(self.episode) + '.png', grasp_imgs)

         # goal images
         dir = "./data/obj_info/obj_info.json"
         with open(dir, mode="rt", encoding="utf-8") as f:
            obj_infos = json.load(f)
         img_num = np.random.randint(1, 10, 1)
         img_name = "rec_goal" if obj_infos["0"]["form"] == "rectangle" else "cir_goal"
         goal_img = cv2.imread("./data/goal/" + img_name + str(img_num[0]) + ".png", cv2.IMREAD_UNCHANGED)

         # place_b images
         place_b_imgs = self.plot_env(episode = self.episode, num_obj = 0, image_state = self.image_states[1])
         actions = self.inference.infer(grasp_imgs[1], goal_img, method, place_images=place_b_imgs[1])
         # TODO: planning grasp_trajectry

         reward = 0
         for obj_info in obj_infos:
            grasp_execute = self.grasp_decision.is_cheked_grasping(actions["grasp"], obj_infos[str(obj_info)])
            if grasp_execute:
               place_obj_info = obj_infos[str(obj_info)]
               reward = 1
               break
         actions["grasp"]["reward"] = reward

         if grasp_execute:
            # TODO:planning place_trajectry
            place_a_imgs = self.plot_env(episode = self.episode, num_obj = 1, image_state = self.image_states[3], action=actions["place"]["pose"], obj_info = place_obj_info)
            reward = 1
            print(" #### place_success #### ")
         else:
            place_a_imgs = self.plot_env(episode = self.episode, num_obj = 0, image_state = self.image_states[3])
            reward = 0
         actions["place"]["reward"] = reward
         
         # save data
         self.dataset[str(self.episode)] = actions
         json_file = open(self.dataset_path, mode="w")
         json.dump(self.dataset, json_file, ensure_ascii=False)
         json_file.close()

         # learning
         if self.episode > 10:
            self.load_train_model = True
         i_time = time.time() - start
         
         start = time.time()
         if self.episode > 5:
            self.train.run(self.load_train_model)

         l_time = time.time() - start

         t_data = [i_time, l_time]
         time_data[str(self.episode)] = t_data
         json_file = open('./data/datasets/main_time.json', mode="w")
         json.dump(time_data, json_file, ensure_ascii=False)
         json_file.close()

         print("inference_time {:.2g}s".format(i_time))
         print("learning_time {:.2g}s".format(l_time))
         print("\n")

         self.episode += 1




parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
parser.add_argument('-e', '--previous_experience', help='Using previous experience', action='store_true')
parser.add_argument('-p', '--dataset_path', action='store', type=str, default='./data/datasets/datasets.json')
parser.add_argument('-m', '--percentage_secondary_method', action='store', type=float, default=0.)
args = parser.parse_args()

learn = SelfLearning(
   previous_experience = args.previous_experience,
   dataset_path = args.dataset_path,
   percentage_secondary = args.percentage_secondary_method,
)
learn.manipulate()