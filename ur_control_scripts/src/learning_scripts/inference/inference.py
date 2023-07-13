import time
from typing import List
import json

import math
import numpy as np
import cv2
from itertools import product
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from learning_scripts.inference.inference_utils import InferenceUtils
from learning_scripts.utils.param import SelectionMethod
from learning_scripts.models.models import GraspModel, PlaceModel, MergeModel


class Inference(InferenceUtils):
   def __init__(self, lower_random_pose=[0., 0., -1.484], upper_random_pose=[480., 752., 1.484], ls_path=None):
      super(Inference, self).__init__(
         lower_random_pose=lower_random_pose,
         upper_random_pose=upper_random_pose,
         ls_path=ls_path
      )
      self.number_top_grasp = 200
      self.number_top_place = 200
      self.input_shape = [None, None, 1] if True else [None, None, 3]
      self.z_shape = 48

      # load model
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.grasp_model = GraspModel(self.input_shape[2]).float().to(self.device)
      self.place_model = PlaceModel(self.input_shape[2]*2).float().to(self.device)
      self.merge_model = MergeModel(self.z_shape).float().to(self.device)
      with open(self.ls_path + '/data/checkpoints/timestamp.txt', 'r') as f:
         self.previous_model_timestanp = f.read()
      self.reload_model_weights()


   def reload_model_weights(self, lode_model = True):
      with open(self.ls_path + '/data/checkpoints/timestamp.txt', 'r') as f:
         saved_timestamp = f.read()
      if lode_model and self.previous_model_timestanp!=saved_timestamp:
         cptfile = self.ls_path + '/data/checkpoints/model.cpt'
         cpt = torch.load(cptfile)
         self.grasp_model.load_state_dict(cpt['grasp_model_state_dict'])
         self.place_model.load_state_dict(cpt['place_model_state_dict'])
         self.merge_model.load_state_dict(cpt['merge_model_state_dict'])
      self.previous_model_timestanp = saved_timestamp


   def infer(
         self,
         images,
         goal_images,
         method: SelectionMethod,
         place_images = None,
      ):
      self.reload_model_weights()
      actions = {}
      grasp_action = {}
      place_action = {}
      if method == SelectionMethod.Random:
         grasp_action["index"] = int(np.random.choice(range(1)))
         grasp_action["pose"] = [np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                                 np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                                 np.random.uniform(self.lower_random_pose[2], self.upper_random_pose[2])] 
         grasp_action["step"] = 0

         place_action["index"] = int(np.random.choice(range(3)))
         place_action["pose"] = [np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                                 np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                                 np.random.uniform(self.lower_random_pose[2], self.upper_random_pose[2])]  # [rad]
         place_action["step"] = 0

         actions["grasp"] = grasp_action
         actions["place"] = place_action

         return actions

      elif method == "oracle":
         dir = self.ls_path + "/data/obj_info/obj_info.json"
         with open(dir, mode="rt", encoding="utf-8") as f:
            obj_infos = json.load(f)
         keys = list(obj_infos.keys())
         obj_info = obj_infos[keys[-1]]
         pose = obj_info["center_psoe"]
         pose[0] += int(np.random.normal(0, 10, 1))
         pose[1] += int(np.random.normal(0, 10, 1))
         if isinstance(obj_info["angle"], type(None)):
            pose.append(np.random.normal(0, math.pi/4, 1)[0])
         else:
            pose.append(obj_info["angle"] + np.random.normal(0, math.pi/6, 1)[0])
         grasp_action["index"] = int(np.random.choice(range(3)))
         grasp_action["pose"] = pose
         grasp_action["step"] = 0

         place_action["index"] = int(np.random.choice(range(3)))
         place_action["pose"] = [np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                                 np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                                 np.random.uniform(self.lower_random_pose[2], self.upper_random_pose[2])]  # [rad]
         place_action["step"] = 0

         actions["grasp"] = grasp_action
         actions["place"] = place_action

         return actions
   
      input_images = self.get_images(images)
      goal_input_images = self.get_images(goal_images)
      place_input_images = self.get_images(place_images)

      input_images = torch.tensor(input_images).to(self.device)
      input_images = torch.permute(input_images, (0, 3, 1, 2)).float()
      goal_input_images = torch.tensor(goal_input_images).to(self.device)
      goal_input_images = torch.permute(goal_input_images, (0, 3, 1, 2)).float()
      place_input_images = torch.tensor(place_input_images).to(self.device)
      place_input_images = torch.permute(place_input_images, (0, 3, 1, 2)).float()

      self.grasp_model.eval()
      self.place_model.eval()
      self.merge_model.eval()

      z_g, reward_g = self.grasp_model(input_images)
      z_p, reward_p = self.place_model(place_input_images, goal_input_images)
      z_g = torch.permute(z_g, (0, 2, 3, 1)).float()
      z_p = torch.permute(z_p, (0, 2, 3, 1)).float()
      reward_g = torch.permute(reward_g, (0, 2, 3, 1)).float()
      reward_p = torch.permute(reward_p, (0, 2, 3, 1)).float()

      first_method = SelectionMethod.PowerProb if method in [SelectionMethod.Top5, SelectionMethod.Max] else method
      # first_method = SelectionMethod.Top5

      filter_lambda_n_grasp = self.get_filter_n(first_method, self.number_top_grasp)
      filter_lambda_n_place = self.get_filter_n(first_method, self.number_top_place)

      np_reward_g = reward_g.to('cpu').detach().numpy().copy()
      np_reward_p = reward_p.to('cpu').detach().numpy().copy()
      g_top_index = filter_lambda_n_grasp(np_reward_g)
      p_top_index = filter_lambda_n_place(np_reward_p)

      g_top_index_unraveled = np.transpose(np.asarray(np.unravel_index(g_top_index, reward_g.shape)))
      p_top_index_unraveled = np.transpose(np.asarray(np.unravel_index(p_top_index, reward_p.shape)))


      g_top_z = z_g[g_top_index_unraveled[:, 0], g_top_index_unraveled[:, 1], g_top_index_unraveled[:, 2]]
      p_top_z = z_p[p_top_index_unraveled[:, 0], p_top_index_unraveled[:, 1], p_top_index_unraveled[:, 2]]

      combined_dataset = CombinedDataset(g_top_z, p_top_z)
      dataloader = DataLoader(combined_dataset, batch_size=200, shuffle=False)


      rewards = torch.empty(0).to(self.device)
      for batch_data1, batch_data2, batch_indices1, batch_indices2 in dataloader:
         reward = self.merge_model([batch_data1.to(self.device), batch_data2.to(self.device)])
         rewards = torch.cat((rewards, torch.unsqueeze(reward, dim=0)), dim=0)
      reward = rewards.to('cpu').detach().numpy().copy()
      
      g_toreward_p = np_reward_g[g_top_index_unraveled[:, 0], g_top_index_unraveled[:, 1], g_top_index_unraveled[:, 2], g_top_index_unraveled[:, 3]]
      g_toreward_p_repeated = np.repeat(np.expand_dims(np.expand_dims(g_toreward_p, axis=1), axis=1), self.number_top_place, axis=1)

      filter_measure = reward * g_toreward_p_repeated

      filter_lambda = self.get_filter(method)
      index_raveled = filter_lambda(filter_measure)


      index_unraveled = np.unravel_index(index_raveled, reward.shape)
      g_index = g_top_index_unraveled[index_unraveled[0]]
      p_index = p_top_index_unraveled[index_unraveled[1]]


      grasp_action["index"] = int(g_index[3])
      grasp_action["pose"] = self.pose_from_index(g_index, reward_g.shape)
      grasp_action["estimated_reward"] = int(reward_g[tuple(g_index)])
      grasp_action["step"] = 0


      place_action["index"] = int(g_index[3])
      place_action["pose"] = self.pose_from_index(p_index, reward_p.shape, resolution_factor=1.0)
      place_action["estimated_reward"] = int(reward[index_unraveled])
      place_action["step"] = 0
   
      actions["grasp"] = grasp_action
      actions["place"] = place_action
      return actions
      

class CombinedDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.combinations = list(product(range(len(data1)), range(len(data2))))

    def __getitem__(self, index):
        index1, index2 = self.combinations[index]
        return self.data1[index1], self.data2[index2], index1, index2

    def __len__(self):
        return len(self.combinations)