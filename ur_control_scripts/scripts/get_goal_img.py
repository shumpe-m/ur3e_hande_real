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
      self.ls_path = "/root/learning_data"
      with open(self.ls_path + '/config/config.yml', 'r') as yml:
         config = yaml.safe_load(yml)
      self.min = config["depth"]["distance_min"]
      self.max = config["depth"]["distance_max"]

   def take_goal_img(self, num=0, step=0):
      self.go_default_pose("backward")
      goal_depth_img, _, _ = self.take_images("depth", self.min, self.max)
      goal_color_img, _, _ = self.take_images("color", self.min, self.max)
      cv2.imwrite(self.ls_path + '/data/goal/depth_goal' + str(num) + str(step) + '.png', goal_depth_img)
      cv2.imwrite(self.ls_path + '/data/goal/color_goal' + str(num) + str(step) + '.png', goal_color_img)

      # im_gray = cv2.imread(self.ls_path + '/data/goal/depth_goal' + str(num) + str(step) + '.png', cv2.IMREAD_UNCHANGED)

      # plt.imshow(im_gray, cmap='gray')
      # plt.show()



parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
parser.add_argument('-p', '--dataset_path', action='store', type=str, default='/root/learning_data/data/datasets/datasets.json')
args = parser.parse_args()

get_img = GetGoalImg(
   dataset_path = args.dataset_path,
)

get_img.take_goal_img()