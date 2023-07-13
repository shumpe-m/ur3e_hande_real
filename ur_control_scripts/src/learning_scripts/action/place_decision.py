import pathlib

import numpy as np
import math

class PlaceDecision:
   def is_cheked_placing(self, action, using = False):
    collision = False
    if using:
        if 480 / 2 - 80 <= action[0] and 480 / 2 + 80 >= action[0] and 752 / 2 - 70 <= action[1] and 752 / 2 + 70 >= action[1]:
            collision = True
       
    return collision