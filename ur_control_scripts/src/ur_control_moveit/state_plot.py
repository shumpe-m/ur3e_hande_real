#!/usr/bin/env python3
# coding: UTF-8
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tf
import geometry_msgs.msg

from motion_capture import transformations

class PlotPose:
    def rot(self, rpy):
        """
        This function calculates a rotation matrix.

        Parameters
        ----------
        rpy : list [roll, pitch, yaw]
            rpy is roll, pitch, and yaw.

        Returns
        -------
        R : numpy.ndarray
            Rotation matrix in 3 dimensions.
        """
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), np.sin(roll)],
                    [0, -np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                    [0, 1, 0],
                    [np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        R = Rz.dot(Ry).dot(Rx)

        return R


    def plot_p(self, target_states, states):
        trf = transformations.Transformations()
        fig,ax= plt.subplots(2,4,figsize=plt.figaspect(1),subplot_kw=dict(projection='3d'))
        ax=ax.ravel()

        # 各次元の変化量を指定
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
                    ax[id].scatter(obs_state[1], obs_state[2], obs_state[3], color = "blue") # 始点


            ax[id].set_xlabel('x') # x軸ラベル
            ax[id].set_ylabel('y') # y軸ラベル
            ax[id].set_zlabel('z') # z軸ラベル
            ax[id].set_title(name + "_scatter", fontsize=12) # タイトル
            ax[id].legend() # 凡例
            ax[id].set_xlim(-0.2, 0.2) # x軸の表示範囲
            ax[id].set_ylim(0.1, 0.5) # y軸の表示範囲
            ax[id].set_zlim(0, 0.4) # z軸の表示範囲

        for id, name in enumerate(target_name):
            
            # target_state plot
            target_index = np.where(target_states[:, 0] == id)
            if target_index[0].size != 0:
                target_state = target_states[target_index[0]]
                target_e = trf.quaternion_to_euler(target_state[0, 4:])
                R_t = self.rot(target_e)
                target_x = R_t.dot(x_arrow)
                target_y = R_t.dot(y_arrow)
                target_z = R_t.dot(z_arrow)
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_x[0], target_x[1], target_x[2], arrow_length_ratio=0.1, color = "red") # 矢印プロット
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_y[0], target_y[1], target_y[2], arrow_length_ratio=0.1, color = "blue") # 矢印プロット
                ax[id + 4].quiver(target_state[0, 1], target_state[0, 2], target_state[0, 3], target_z[0], target_z[1], target_z[2], arrow_length_ratio=0.1, color = "green") # 矢印プロット
                ax[id + 4].scatter(target_state[0, 1], target_state[0, 2], target_state[0, 3], color = "black") # 始点

                # observation state plot
                obs_index = np.where(states[:, 0] == id)
                for odx in obs_index[0]:
                    obs_state = states[odx]
                    obs_e = trf.quaternion_to_euler(obs_state[4:])
                    R_o = self.rot(obs_e)
                    obs_x = R_o.dot(x_arrow)
                    obs_y = R_o.dot(y_arrow)
                    obs_z = R_o.dot(z_arrow)
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_x[0], obs_x[1], obs_x[2], arrow_length_ratio=0.1, color = "gold") # 矢印プロット
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_y[0], obs_y[1], obs_y[2], arrow_length_ratio=0.1, color = "cyan") # 矢印プロット
                    ax[id + 4].quiver(obs_state[1], obs_state[2], obs_state[3], obs_z[0], obs_z[1], obs_z[2], arrow_length_ratio=0.1, color = "lime") # 矢印プロット
                    ax[id + 4].scatter(obs_state[1], obs_state[2], obs_state[3], color = "gray") # 始点


            ax[id + 4].set_xlabel('x') # x軸ラベル
            ax[id + 4].set_ylabel('y') # y軸ラベル
            ax[id + 4].set_zlabel('z') # z軸ラベル
            ax[id + 4].set_title(name + "_tf", fontsize=20) # タイトル
            ax[id + 4].legend() # 凡例
            ax[id + 4].set_xlim(-0.2, 0.2) # x軸の表示範囲
            ax[id + 4].set_ylim(0.1, 0.5) # y軸の表示範囲
            ax[id + 4].set_zlim(0, 0.4) # z軸の表示範囲
        plt.show()



