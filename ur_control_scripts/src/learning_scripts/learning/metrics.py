import torch
import torch.nn.functional as F


class Split:
    confident_learning_alpha = None  # in [0, 1]

    @classmethod
    def single_class_split(cls, y_pred, y_true, device):
        value_true = y_true[:, 0]
        index = y_true[:, 1, 0].clone().detach().to(torch.int64).unsqueeze(dim=1)
        sample_weight = y_true[:, 2]

        grasp_pred = torch.gather(y_pred, 1, index)
        value_pred = torch.cat([grasp_pred, y_pred[:,-2:]], dim=1)

        return value_true, value_pred, sample_weight

        
class Losses:
    def __init__(self, device):
        self.device = device

    def binary_crossentropy(self, y_pred, y_true):
        value_true, value_pred, sample_weight = Split.single_class_split(y_pred, y_true, self.device)

        loss = torch.nn.BCELoss(weight = sample_weight)
        return loss(value_pred, value_true)