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


class GetGoalImg(UrControl):
   def __init__(self, dataset_path):
      super().__init__()
      self.dataset_path = dataset_path
      self.ls_path = "/root/catkin_ws/src/ur3e_hande_real/ur_control_scripts/src/learning_scripts"
      with open(self.ls_path + '/config/config.yml', 'r') as yml:
         config = yaml.safe_load(yml)
      self.min = config["depth"]["distance_min"]
      self.max = config["depth"]["distance_max"]

   def take_goal_img(self, num=0, step=0):
      img_type = "depth"
      self.go_default_pose("backward")
      goal_img, _, _ = self.take_images(img_type, self.min, self.max)
      cv2.imwrite(self.ls_path + '/data/goal/depth_goal' + str(num) + str(step) + '.png', goal_img)

      im_gray = cv2.imread(self.ls_path + '/data/goal/depth_goal' + str(num) + str(step) + '.png', cv2.IMREAD_UNCHANGED)

      # plt.imshow(im_gray, cmap='gray')
      # plt.show()



parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
parser.add_argument('-p', '--dataset_path', action='store', type=str, default='/root/catkin_ws/src/ur3e_hande_real/ur_control_scripts/src/learning_scripts/data/datasets/datasets.json')
args = parser.parse_args()

get_img = GetGoalImg(
   dataset_path = args.dataset_path,
)

get_img.take_goal_img()