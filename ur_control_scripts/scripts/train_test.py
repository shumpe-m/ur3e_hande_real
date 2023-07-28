#!/usr/bin/python3
from manipulate import UrControl
from learning_scripts.learning.train import Train

train = Train(
   dataset_path='/root/learning_data/data/datasets/datasets.json',
   ls_path="/root/learning_data",
   image_format="png",
)

train.run(False)