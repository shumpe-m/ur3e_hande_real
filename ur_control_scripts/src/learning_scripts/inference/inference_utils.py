import yaml

import copy
import numpy as np
import cv2 

from learning_scripts.utils.param import SelectionMethod

class InferenceUtils:
   def __init__(self,
               lower_random_pose=200,
               upper_random_pose=200,
               monte_carlo=None,
               input_uncertainty=False,
               with_types=False,
               ls_path=None,
   ):
      self.ls_path = ls_path
      with open(self.ls_path + '/config/config.yml', 'r') as yml:
         config = yaml.safe_load(yml)
      self.size_input = (config["inference"]["img_width"], config["inference"]["img_height"])
      self.size_original_cropped = (config["inference"]["size_original_cropped"], config["inference"]["size_original_cropped"])
      self.size_output = (config["inference"]["size_output"], config["inference"]["size_output"])
      self.size_cropped = (config["inference"]["size_cropped"], config["inference"]["size_cropped"])
      self.reward_g_shape = (None, config["inference"]["reward_g_shape"], config["inference"]["reward_g_shape"], None)
      self.scale_factors = (
         float(self.size_original_cropped[0]) / float(self.size_output[0]),
         float(self.size_original_cropped[1]) / float(self.size_output[1])
      )

      self.a_space = np.linspace(-np.pi / 2, np.pi / 2, 16)

      self.lower_random_pose = lower_random_pose
      self.upper_random_pose = upper_random_pose


   def pose_from_index(self, index, index_shape, resolution_factor=2.0):
      size_reward_center = (index_shape[1] / 2, index_shape[2] / 2)
      scale = self.size_original_cropped[0] / self.size_output[0] * ((self.reward_g_shape[1] * 2 - 1) / index_shape[1])

      rot_mat = cv2.getRotationMatrix2D(size_reward_center, -self.a_space[index[0]] * 180.0 / np.pi, scale)
      rot_mat[0][2] += self.size_input[0] / 2 - size_reward_center[0]
      rot_mat[1][2] += self.size_input[1] / 2 - size_reward_center[1]

      index_xy1 = np.array([index[1], index[2], 1.0]) 
      xy = np.dot(rot_mat, index_xy1)
      xy[0] = np.clip(xy[0], 0, 480)
      xy[1] = np.clip(xy[1], 0, 752)
      x = xy[0]
      y = xy[1]
      a = -self.a_space[index[0]]  # [rad]

      return [x, y, a]


   def get_images(self, orig_image):
      image = copy.deepcopy(orig_image)

      mat_images = []
      for a in self.a_space:
         rot_mat = cv2.getRotationMatrix2D((self.size_input[0] / 2, self.size_input[1] / 2),
                                          a * 180.0 / np.pi,
                                          scale=self.size_output[0] / self.size_original_cropped[0],)
         rot_mat[:, 2] += [(self.size_cropped[0] - self.size_input[0]) / 2,
                           (self.size_cropped[1] - self.size_input[1]) / 2]     
         dst_depth = cv2.warpAffine(image, rot_mat, self.size_cropped, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
         mat_images.append(dst_depth)

      # mat_images = np.array(mat_images) / np.iinfo(orig_image.dtype).max
      mat_images = np.array(mat_images)
      if len(mat_images.shape) == 3:
         mat_images = np.expand_dims(mat_images, axis=-1)

      return mat_images


   @classmethod
   def get_filter(cls, method: SelectionMethod, n=5):
      if method == SelectionMethod.Top5:
         return lambda x: np.random.choice(np.argpartition(x, -n, axis=None)[-n:])
      if method == SelectionMethod.Uncertain:
         return lambda x: np.argmin(np.abs(x - 0.5))
      if method == SelectionMethod.RandomInference:
         return lambda x: np.random.choice(np.arange(x.size))
      if method == SelectionMethod.NotZero:
         return lambda x: np.random.choice(np.flatnonzero(x >= min(0.05, np.amax(x))))
      if method == SelectionMethod.Prob:
         return lambda x: np.random.choice(np.arange(x.size), p=(np.ravel(x) / np.sum(np.ravel(x))))
      if method == SelectionMethod.PowerProb:
         return lambda x: np.random.choice(np.arange(x.size), p=(np.power(np.ravel(x), 6) / np.sum(np.power(np.ravel(x), 6))))
      if method == SelectionMethod.Max:
         return lambda x: x.argmax()
      if method == SelectionMethod.Min:
         return lambda x: x.argmin()
      raise Exception(f'Selection method not implemented: {method}')

   
   @classmethod
   def get_filter_n(cls, method: SelectionMethod, n: int):
      if method == SelectionMethod.Top5 or method == SelectionMethod.Max:
         return lambda x: np.argpartition(x, -n, axis=None)[-n:]
      if method == SelectionMethod.RandomInference:
         return lambda x: np.random.choice(np.arange(x.size), size=n)
      if method == SelectionMethod.Prob:
         return lambda x: np.random.choice(np.arange(x.size), size=n, p=(np.ravel(x) / np.sum(np.ravel(x))), replace=False)
      if method == SelectionMethod.PowerProb:
         def power_prob(x):
               p_x = np.power(np.ravel(x), 6)
               return np.random.choice(np.arange(x.size), size=n, p=p_x / np.sum(p_x), replace=False)
         return power_prob
      if method == SelectionMethod.ExpProb:
         def exp_prob(x):
               p_x = np.exp(-5 * (1 / np.ravel(x) - 1))
               return np.random.choice(np.arange(x.size), size=n, p=p_x / np.sum(p_x), replace=False)
         return exp_prob
      raise Exception(f'Selection method for N not implemented: {method}')