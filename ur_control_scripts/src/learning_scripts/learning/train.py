#!/usr/bin/python3
import datetime
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learning_scripts.models.models import GraspModel, PlaceModel, MergeModel, Combined_model
from learning_scripts.learning.datasets import CustomDataset
from learning_scripts.learning.metrics import Losses


class Train:
   def __init__(self, dataset_path=None, image_format='png'):
      self.input_shape = [None, None, 1] if True else [None, None, 3]
      self.z_shape = 48
      self.train_batch_size = 512
      self.validation_batch_size = 256
      self.percent_validation_set = 0.2
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      torch.manual_seed(0)

      self.previous_epoch = 0
      self.dataset_path = dataset_path

      self.dataset_tensor = []

   def run(self, load_model=True):
      time_data = {}
      with open("./data/datasets/learning_time.json", mode="rt", encoding="utf-8") as f:
         time_data = json.load(f)

      # get dataset
      start = time.time()
      with open(self.dataset_path, mode="rt", encoding="utf-8") as f:
         all_data = json.load(f)
      custom_ds = CustomDataset(all_data, seed=42)
      datasets = custom_ds.get_data(self.dataset_tensor)
      datasets_length = len(datasets)
      val_data_length = int(datasets_length * self.percent_validation_set)
      train_data_length = datasets_length - val_data_length
      train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_data_length, val_data_length])

      train_dataloaders = DataLoader(train_dataset, 
                                    batch_size=self.train_batch_size,
                                    shuffle=True,
                                    num_workers=0, 
                                    drop_last=False,
                                    pin_memory=True
                                    )

      val_dataloader = DataLoader(val_dataset, 
                                 batch_size=self.validation_batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=False,
                                 pin_memory=True)
                                 
      self.dataset_tensor = datasets
      dataset_time = time.time() - start

      start = time.time()
      # set up nn model
      grasp_model = GraspModel(self.input_shape[2]).to(self.device)
      place_model = PlaceModel(self.input_shape[2]*2).to(self.device)
      merge_model = MergeModel(self.z_shape).to(self.device)
      model = Combined_model(grasp_model, place_model, merge_model).to(self.device)

      # optimizer
      reward_param = []
      z_param = []
      merge_param = []
      other_param = []
      for name, param in model.named_parameters():
         if '_r_last' in name:
            reward_param.append(param)
         elif '_z_last' in name:
            z_param.append(param)
         elif 'merge_model.linear_block' in name:
            merge_param.append(param)
         else:
            other_param.append(param)
      optimizer = torch.optim.Adam([
         {'params': reward_param, 'weight_decay': 0.0},
         {'params': z_param, 'weight_decay': 0.0005},
         {'params': merge_param, 'weight_decay': 0.01},
         {'params': other_param, 'weight_decay': 0.001},
      ], lr=2e-4)

      # loss function
      criterion = Losses(self.device)

      # load model
      if load_model:
         cptfile = '/root/2D-sim/scripts/data/checkpoints/out.cpt'
         cpt = torch.load(cptfile)
         stdict_m = cpt['combined_model_state_dict']
         stdict_o = cpt['opt_state_dict']
         model.load_state_dict(stdict_m)
         optimizer.load_state_dict(stdict_o)
      model_time = time.time() - start

      start = time.time()
      self.train(train_dataloaders, model, criterion, optimizer)
      train_time = time.time() - start

      start = time.time()
      self.test(val_dataloader, model, criterion, optimizer)
      val_time = time.time() - start

      time_data[str(len(time_data))] = [dataset_time, train_time, val_time, model_time]
      json_file = open('./data/datasets/learning_time.json', mode="w")
      json.dump(time_data, json_file, ensure_ascii=False)
      json_file.close()




   def train(self, dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      losses = []
      train_loss= 0
      for x, y in dataloader:
         x = tuple(torch.reshape(x_arr, (-1, 1, 32, 32)).to(self.device) for x_arr in x)
         y = tuple(torch.reshape(y_arr, (-1, 3)).to(self.device) for y_arr in y)
         y = torch.cat([y[0], y[1], y[2]], dim=1).view(-1, 3, 3)
         if x[0].shape[0] == 1:
            break
         z_g, reward_g, z_p, reward_p, reward = model(x[0],x[1],x[2])
         pred = torch.cat([reward_g, reward_p, reward], dim=1)
         loss = loss_fn.binary_crossentropy(pred, y)

         # Backpropagation
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         losses.append(loss.item())
         train_loss += loss.item()


      outfile = '/root/2D-sim/scripts/data/checkpoints/out.cpt'
      torch.save({'combined_model_state_dict': model.state_dict(),
                  'grasp_model_state_dict': model.grasp_model.state_dict(),
                  'place_model_state_dict': model.place_model.state_dict(),
                  'merge_model_state_dict': model.merge_model.state_dict(),
                  'opt_state_dict': optimizer.state_dict(),
                  'loss': losses,
                  }, outfile)
      timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      with open('/root/2D-sim/scripts/data/checkpoints/timestamp.txt', 'w') as f:
         f.write(timestamp)


   def test(self, dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
         for x, y in dataloader:
            if x[0].shape[0] == 1:
               break
            x = tuple(torch.reshape(x_arr, (-1, 1, 32, 32)).to(self.device) for x_arr in x)
            y = tuple(torch.reshape(y_arr, (-1, 3)).to(self.device) for y_arr in y)
            y = torch.cat([y[0], y[1], y[2]], dim=1).view(-1, 3, 3)

            z_g, reward_g, z_p, reward_p, reward = model(x[0],x[1],x[2])
            test_loss += loss_fn.binary_crossentropy(torch.cat([reward_g, reward_p, reward], dim=1), y).item()

      test_loss /= size
      print(f"Avg loss: {test_loss:>8f}")
