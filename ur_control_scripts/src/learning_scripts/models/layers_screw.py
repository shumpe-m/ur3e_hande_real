import torch
import torch.nn as nn



# Custum Layer
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, dropout_rate = 0.42):
      super(ConvBlock, self).__init__()
      self.conv = nn.Conv2d(in_channels = in_channels, 
                           out_channels = out_channels, 
                           kernel_size = kernel_size, 
                           stride = stride)
      self.lrelu = nn.LeakyReLU()
      self.b_norm = nn.BatchNorm2d(num_features = out_channels)
      self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, x):
      x = self.conv(x)
      x = self.lrelu(x)
      x = self.b_norm(x)
      x = self.dropout(x)

      return x


class ConvBlockRemoveWeightNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):
      super(ConvBlockRemoveWeightNorm, self).__init__()
      self.conv = nn.Conv2d(in_channels = in_channels, 
                                             out_channels = out_channels, 
                                             kernel_size = kernel_size, 
                                             stride = stride)

    def forward(self, x):
      x = self.conv(x)

      return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias = True, dropout_rate = 0.42):
      super(LinearBlock, self).__init__()
      self.linear = nn.Linear(in_features = in_features, 
                           out_features = out_features, 
                           bias = bias)
      self.lrelu = nn.LeakyReLU()
      self.b_norm = nn.BatchNorm1d(num_features = out_features)
      self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, x):
      x = self.linear(x)
      x = self.lrelu(x)
      x = self.b_norm(x)
      x = self.dropout(x)

      return x


class LinearBlockRemoveWeightNorm(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
      super(LinearBlockRemoveWeightNorm, self).__init__()
      self.linear = nn.Linear(in_features = in_features, 
                                                out_features = out_features, 
                                                bias = bias)
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      x = self.linear(x)
      x = self.sigmoid(x)

      return x