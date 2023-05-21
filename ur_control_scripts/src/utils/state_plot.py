#!/usr/bin/env python3
# coding: UTF-8
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tf
import geometry_msgs.msg

from utils import transformations

class PlotPose:
    def plot_p(self, target_states, states):
        trf = utils.Transformations()
        fig,ax= plt.subplots(2,4,figsize=plt.figaspect(1),subplot_kw=dict(projection='3d'))
        ax=ax.ravel()

        # arrow length
        len_ax = 0.1
        x_arrow = [len_ax,0,0]
        y_arrow = [0,len_ax,0]
        z_arrow = [0,0,len_ax]

        target_name = ["Chikuwa", "Shrimp", "Eggplant", "Green_papper"]


        for id, name in enumerate(target_name):
            
            # target_state plot
            target_index = np.where(target_states[:, 0] == id)
            if target_index[0].size != 0:
                target_state = target_states[target_index[0]]
                ax[id].scatter(target_state[0, 1], target_state[0, 2], target_state[0, 3], color = "red") # 始点

                # observation state plot
                obs_index = np.where(states[:, 0] == id)
                for odx in obs_index[0]:
                    obs_state = states[odx]
                    # draw pose positions
                    ax[id].scatter(obs_state[1], obs_state[2], obs_state[3], color = "blue") # 始点

            # set label
            ax[id].set_xlabel('x')
            ax[id].set_ylabel('y')
            ax[id].set_zlabel('z')
            # set title
            ax[id].set_title(name + "_scatter", fontsize=12)
            # set legend
            ax[id].legend()
            # set axis limit
            ax[id].set_xlim(-0.2, 0.2)
            ax[id].set_ylim(0.1, 0.5)
            ax[id].set_zlim(0, 0.4)

        for id, name in enumerate(target_name):
            
            # target_state plot
            target_index = np.where(target_states[:, 0] == id)
            if target_index[0].size != 0:
                target_state = target_states[target_index[0]]
                target_e = trf.quaternion_to_euler(target_state[0, 4:])
                R_t = trf.rot(target_e)
                target_x = R_t.dot(x_arrow)
                target_y = R_t.dot(y_arrow)
                target_z = R_t.dot(z_arrow)
                # draw pose positions and orientations
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_x[0], target_x[1], target_x[2], arrow_length_ratio=0.15, color = "red")
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_y[0], target_y[1], target_y[2], arrow_length_ratio=0.15, color = "blue")
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_z[0], target_z[1], target_z[2], arrow_length_ratio=0.15, color = "green")
                ax[id + 4].scatter(target_state[0, 1], target_state[0, 2], target_state[0, 3], color = "black")

                # observation state plot
                obs_index = np.where(states[:, 0] == id)
                for odx in obs_index[0]:
                    obs_state = states[odx]
                    obs_e = trf.quaternion_to_euler(obs_state[4:])
                    R_o = trf.rot(obs_e)
                    obs_x = R_o.dot(x_arrow)
                    obs_y = R_o.dot(y_arrow)
                    obs_z = R_o.dot(z_arrow)
                    # draw pose positions and orientations
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_x[0], obs_x[1], obs_x[2], arrow_length_ratio=0.15, color = "gold")
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_y[0], obs_y[1], obs_y[2], arrow_length_ratio=0.15, color = "cyan")
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_z[0], obs_z[1], obs_z[2], arrow_length_ratio=0.15, color = "lime")
                    ax[id + 4].scatter(obs_state[1], obs_state[2], obs_state[3], color = "gray")


            ax[id + 4].set_xlabel('x')
            ax[id + 4].set_ylabel('y')
            ax[id + 4].set_zlabel('z')
            ax[id + 4].set_title(name + "_tf", fontsize=20)
            ax[id + 4].legend()
            ax[id + 4].set_xlim(-0.2, 0.2)
            ax[id + 4].set_ylim(0.1, 0.5)
            ax[id + 4].set_zlim(0, 0.4)
        plt.show()



