import torch
import torch.nn as nn

from learning_scripts.models.layers import ConvBlock, LinearBlock

class GraspModel(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      self.sigmoid = nn.Sigmoid()

      self.conv_block1 = ConvBlock(in_channels = in_channels, out_channels = 32)
      self.conv_block2 = ConvBlock(in_channels = 32, out_channels = 32, stride = 2)
      self.conv_block3 = ConvBlock(in_channels = 32, out_channels = 32)

      self.conv_block_r1 = ConvBlock(in_channels = 32, out_channels = 48)
      self.conv_block_r2 = ConvBlock(in_channels = 48, out_channels = 48)
      
      self.conv_block_r3 = ConvBlock(in_channels = 48, out_channels = 64)
      self.conv_block_r4 = ConvBlock(in_channels = 64, out_channels = 64)
   
      self.conv_block_r5 = ConvBlock(in_channels = 64, out_channels = 64)
      self.conv_block_r6 = ConvBlock(in_channels = 64, out_channels = 48, kernel_size = 2)
      self.conv_block_r_last = nn.Conv2d(in_channels = 48, out_channels = 4, kernel_size = 1, stride = 1)

      self.conv_block_z1 = ConvBlock(in_channels = 32, out_channels = 64)
      self.conv_block_z2 = ConvBlock(in_channels = 64, out_channels = 64)
      
      self.conv_block_z3 = ConvBlock(in_channels = 64, out_channels = 96)
      self.conv_block_z4 = ConvBlock(in_channels = 96, out_channels = 96)
   
      self.conv_block_z5 = ConvBlock(in_channels = 96, out_channels = 128)
      self.conv_block_z6 = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 2)
      self.conv_block_z_last = nn.Conv2d(in_channels = 128, out_channels = 48, kernel_size = 1)

    def forward(self, inputs):
      x = self.conv_block1(inputs)
      x = self.conv_block2(x)
      x = self.conv_block3(x)

      x_r = self.conv_block_r1(x)
      x_r = self.conv_block_r2(x_r)

      x_r = self.conv_block_r3(x_r)
      x_r = self.conv_block_r4(x_r)

      x_r = self.conv_block_r5(x_r)
      x_r = self.conv_block_r6(x_r)
      reward = self.conv_block_r_last(x_r)
      reward = self.sigmoid(reward)

      x = self.conv_block_z1(x)
      x = self.conv_block_z2(x)

      x = self.conv_block_z3(x)
      x = self.conv_block_z4(x)

      x = self.conv_block_z5(x)
      x = self.conv_block_z6(x)
      z = self.conv_block_z_last(x)

      return z, reward


class PlaceModel(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      self.sigmoid = nn.Sigmoid()

      self.conv_block1 = ConvBlock(in_channels = in_channels, out_channels = 32)
      self.conv_block2 = ConvBlock(in_channels = 32, out_channels = 32)

      self.conv_block3 = ConvBlock(in_channels = 32, out_channels = 32)
      self.conv_block4 = ConvBlock(in_channels = 32, out_channels = 32)
      self.conv_block5 = ConvBlock(in_channels = 32, out_channels = 32)
      self.conv_block6 = ConvBlock(in_channels = 32, out_channels = 32)

      self.conv_block_r1 = ConvBlock(in_channels = 32, out_channels = 32)
      self.conv_block_r2 = ConvBlock(in_channels = 32, out_channels = 32)
      
      self.conv_block_r3 = ConvBlock(in_channels = 32, out_channels = 48)
      self.conv_block_r4 = ConvBlock(in_channels = 48, out_channels = 48)
      self.conv_block_r5 = ConvBlock(in_channels = 48, out_channels = 48)
      self.conv_block_r6 = ConvBlock(in_channels = 48, out_channels = 48)

      self.conv_block_r7 = ConvBlock(in_channels = 48, out_channels = 48)
      self.conv_block_r8 = ConvBlock(in_channels = 48, out_channels = 48)

      self.conv_block_r9 = ConvBlock(in_channels = 48, out_channels = 64)
      self.conv_block_r10 = ConvBlock(in_channels = 64, out_channels = 48, kernel_size = 2)
      self.conv_block_r_last = nn.Conv2d(in_channels = 48, out_channels = 1, kernel_size = 1, stride = 1)

      self.conv_block_z1 = ConvBlock(in_channels = 32, out_channels = 48)
      self.conv_block_z2 = ConvBlock(in_channels = 48, out_channels = 48)
      
      self.conv_block_z3 = ConvBlock(in_channels = 48, out_channels = 64)
      self.conv_block_z4 = ConvBlock(in_channels = 64, out_channels = 64)
      self.conv_block_z5 = ConvBlock(in_channels = 64, out_channels = 64)
      self.conv_block_z6 = ConvBlock(in_channels = 64, out_channels = 64)

      self.conv_block_z7 = ConvBlock(in_channels = 64, out_channels = 96)
      self.conv_block_z8 = ConvBlock(in_channels = 96, out_channels = 96)

      self.conv_block_z9 = ConvBlock(in_channels = 96, out_channels = 128)
      self.conv_block_z10 = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 2)
      self.conv_block_z_last = nn.Conv2d(in_channels = 128, out_channels = 48, kernel_size = 1)

    def forward(self, inputs1, inputs2):
      x = torch.cat((inputs1, inputs2), 1)

      x = self.conv_block1(x)
      x = self.conv_block2(x)

      x = self.conv_block3(x)
      x = self.conv_block4(x)
      x = self.conv_block5(x)
      x = self.conv_block6(x)

      x_r = self.conv_block_r1(x)
      x_r = self.conv_block_r2(x_r)

      x_r = self.conv_block_r3(x_r)
      x_r = self.conv_block_r4(x_r)
      x_r = self.conv_block_r5(x_r)
      x_r = self.conv_block_r6(x_r)

      x_r = self.conv_block_r7(x_r)
      x_r = self.conv_block_r8(x_r)

      x_r = self.conv_block_r9(x_r)
      x_r = self.conv_block_r10(x_r)
      reward = self.conv_block_r_last(x_r)
      reward = self.sigmoid(reward)

      x = self.conv_block_z1(x)
      x = self.conv_block_z2(x)

      x = self.conv_block_z3(x)
      x = self.conv_block_z4(x)

      x = self.conv_block_z5(x)
      x = self.conv_block_z6(x)
      x = self.conv_block_z7(x)
      x = self.conv_block_z8(x)
      x = self.conv_block_z9(x)
      x = self.conv_block_z10(x)
      z = self.conv_block_z_last(x)

      return z, reward


class MergeModel(nn.Module):
    def __init__(self, in_features):
      super().__init__()
      self.sigmoid = nn.Sigmoid()

      self.linear_block1 = LinearBlock(in_features = in_features, out_features = 128)
      self.linear_block2 = LinearBlock(in_features = 128, out_features = 128)
      self.linear_block3 = LinearBlock(in_features = 128, out_features = 64)
      self.linear_block_r_last = nn.Linear(in_features = 64, out_features = 1)

    def forward(self, inputs):
      x = inputs[0] - inputs[1]

      x = self.linear_block1(x)
      x = self.linear_block2(x)
      x = self.linear_block3(x)
      reward = self.linear_block_r_last(x)
      reward = self.sigmoid(reward)

      return reward

# https://stackoverflow.com/questions/71364119/how-to-combine-two-trained-models-using-pytorch
class Combined_model(nn.Module):
    def __init__(self, grasp_model, place_model, merge_model):
        super(Combined_model, self).__init__()
        self.grasp_model = grasp_model
        self.place_model = place_model
        self.merge_model = merge_model
        
    def forward(self, x1, x2, x3):
        z_g, reward_g = self.grasp_model(x1)
        z_g = torch.reshape(z_g, (z_g.shape[0], z_g.shape[1]))
        reward_g = torch.reshape(reward_g, (reward_g.shape[0], reward_g.shape[1]))
        z_p, reward_p = self.place_model(x2, x3)
        z_p = torch.reshape(z_p, (z_p.shape[0], z_p.shape[1]))
        reward_p = torch.reshape(reward_p, (reward_p.shape[0], reward_p.shape[1]))
        reward = self.merge_model([z_g, z_p])

        return z_g, reward_g, z_p, reward_p, reward