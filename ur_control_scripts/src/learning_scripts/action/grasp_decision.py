import pathlib

import numpy as np
import math

class GraspDecision:
   def is_cheked_grasping(self, action, obj_info):
      distance = np.linalg.norm(action["pose"][:2] - np.array(obj_info["center_psoe"]))
      if isinstance(obj_info["angle"], type(None)):
         a_succesee = True
      else:
         obj_angle = obj_info["angle"]
         angle_diff = abs(action["pose"][2]) - abs(obj_angle)
         a_succesee = abs(angle_diff) <= math.pi/6

      execute = True if distance <= 30 and a_succesee else False
      # print("action:"+str(action)+"  target:"+str(obj_info["center_psoe"])+str(obj_info["angle"]) + "  distance:"+str(distance)+"  :"+str(angle_diff))
      return execute